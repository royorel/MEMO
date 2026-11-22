import os
import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from memo.scene.cameras import Camera
from tqdm import tqdm


def collate_camera(camera_batch):
    if len(camera_batch) == 1:
        return camera_batch[0]
    else:
        return camera_batch


def _camera_from_record(record, image, **kwargs):
    return Camera(
        colmap_id=record['colmap_id'],
        R=record['R'], T=record['T'],
        FoVx=record['FoVx'], FoVy=record['FoVy'],
        image=image,
        image_name=record['image_name'],
        uid=record['uid'],
        normalized_t=record['normalized_t'],
        t=record['t'],
        view_num=record['view_num'],
        cx=record.get('cx'), cy=record.get('cy'),
        data_device='cpu',
        **kwargs,
    )


def camera_from_metadata(meta):
    """The CPU camera of an LMDB record's metadata, with a 1x1 dummy image.

    The true image_width / image_height are passed to the constructor so the
    projection is built at the real resolution.
    """
    return _camera_from_record(meta, torch.zeros(3, 1, 1),
                               image_width=meta['image_width'],
                               image_height=meta['image_height'])


def cached_cameras(dataset):
    """The cameras of a Views4D dataset in key order, built by
    camera_from_metadata.

    Reads the camera metadata from `camera_cache.pt` beside the LMDB when it
    exists. Otherwise reads it from the LMDB records (slow, deserializes the
    images too) and writes that cache file.
    """
    split = "train" if "train" in dataset.lmdb_path else "test"
    cache_file = os.path.join(dataset.lmdb_path, "camera_cache.pt")

    if dataset.camera_cache is not None:
        cache_data = dataset.camera_cache
        desc = f"Loading {split} camera cache"
    else:
        dataset._init_db()
        cache_data = []
        with dataset.env.begin(write=False) as txn:
            for key in tqdm(dataset.views, desc=f"Caching {split} camera poses (first run)", leave=False):
                raw = pickle.loads(txn.get(key.encode()))
                meta = {k: raw[k] for k in (
                    'colmap_id', 'R', 'T', 'FoVx', 'FoVy', 'image_name',
                    'uid', 'normalized_t', 't', 'view_num', 'image_width', 'image_height',
                )}
                if 'cx' in raw:           # per-camera principal point; absent -> centered
                    meta['cx'], meta['cy'] = raw['cx'], raw['cy']
                cache_data.append((key, meta))
        dataset.env.close()
        dataset.env = None
        torch.save(cache_data, cache_file)
        print(f"[INFO] Saved camera cache: {cache_file} ({len(cache_data)} cameras)")
        desc = f"Building {split} cameras"

    cameras = []
    for key, meta in tqdm(cache_data, desc=desc, leave=False):
        cam = camera_from_metadata(meta)
        cam._lmdb_path = dataset.lmdb_path
        cam._lmdb_key = key
        cameras.append(cam)
    return cameras


class ViewpointLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=collate_camera,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor=2,
                 persistent_workers=False, pin_memory_device=''):
        super().__init__(dataset, batch_size, shuffle, sampler,
                         batch_sampler, num_workers, collate_fn,
                         pin_memory, drop_last, timeout,
                         worker_init_fn, multiprocessing_context,
                         generator, prefetch_factor=prefetch_factor if num_workers > 0 else None,
                         persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)
        # Lightweight cameras of all views, kept on CPU to avoid GPU memory
        # fragmentation.
        self._copy_cache = cached_cameras(dataset)

    def __next__(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()

        try:
            return self._iterator.__next__()
        except StopIteration:
            self._iterator = self._get_iterator()
            self._iterator._reset(self)
            return self._iterator.__next__()

    def copy(self):
        """Return a shallow copy of the cached camera list (safe for pop-based sampling)."""
        return list(self._copy_cache)

    def __getitem__(self, idx):
        """Index into the cached camera list (used by training_report)."""
        return self._copy_cache[idx]

    def __len__(self):
        return len(self._copy_cache)


class Views4D(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.camera_cache = None
        cache_file = os.path.join(lmdb_path, "camera_cache.pt")
        if os.path.exists(cache_file):
            # The camera cache lists every view's key in the LMDB's key order,
            # which spares scanning the keys of a multi-gigabyte LMDB, slow on a
            # network filesystem.
            self.camera_cache = torch.load(cache_file, weights_only=False)
            self.views = [key for key, _ in self.camera_cache]
            return
        self._init_db()
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            self.views = [key.decode() for key in cursor.iternext(keys=True, values=False)
                          if key.decode().startswith('frame_')]
        self.env.close()
        self.env = None

    def _init_db(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            if not self.env:
                raise IOError('Cannot open lmdb dataset', self.lmdb_path)

    def __getitem__(self, index):
        self._init_db()
        with self.env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(self.views[index].encode()))
        gt_image = (torch.from_numpy(raw['image']).float() / 255.0).permute(2, 0, 1).contiguous()
        return _camera_from_record(raw, gt_image)

    def __len__(self):
        return len(self.views)
