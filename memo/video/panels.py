"""The mesh panels of the videos rendered with nvdiffrast, and the `full`
layout. The shaded panel is a Lambertian headlamp shading of the mesh, and
the normal panel a matcap of its view-space normals, both double-sided, so
orientation noise from the extraction cannot flip a patch dark."""
import numpy as np
import torch
from PIL import Image

from memo.scene.mesh import Meshes
from memo.utils.graphics_utils import getProjectionMatrix


def set_near(camera, znear):
    """Move the camera's near plane, which nvdiffrast clips the mesh at."""
    device = camera.world_view_transform.device
    camera.znear = znear
    camera.projection_matrix = getProjectionMatrix(
        znear=znear, zfar=camera.zfar, fovX=camera.FoVx, fovY=camera.FoVy,
        cx=camera.cx, cy=camera.cy,
        W=camera.image_width, H=camera.image_height,
    ).transpose(0, 1).to(device)
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0)
        .bmm(camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)


@torch.no_grad()
def mesh_normals(mesh, camera, renderer):
    """World-space face normals of a trimesh mesh seen from `camera`, HxWx3,
    zero where no face is hit."""
    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device="cuda")
    # MeshRenderer interpolates its vertex features unconditionally, and the
    # interpolation rejects zero channels, so the unused depth gives it one.
    rendered = renderer(Meshes(verts=verts, faces=faces).to("cuda"), cameras=[camera], cam_idx=0,
                        return_depth=True, return_normals=True,
                        use_antialiasing=False, check_errors=False)
    return rendered["normals"][0]


def _view_normals(world_normals, camera):
    R_w2c = torch.as_tensor(camera.R, dtype=torch.float32, device=world_normals.device).t()
    return world_normals.reshape(-1, 3) @ R_w2c.t()


def _on_background(rgb, world_normals, background_rgb):
    bg = torch.tensor(background_rgb, dtype=torch.float32, device=world_normals.device)
    has_mesh = world_normals.abs().sum(dim=-1, keepdim=True) > 1e-6
    rgb = torch.where(has_mesh, rgb, bg.expand_as(rgb))
    return (rgb.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()


def shade_lambertian(world_normals, camera, background_rgb):
    """Gray headlamp shading, 0.2 + 0.6 |n_view . z|, over the background."""
    H, W, _ = world_normals.shape
    intensity = _view_normals(world_normals, camera)[:, 2].abs().clamp(max=1.0)
    shaded = (0.2 + 0.6 * intensity).reshape(H, W, 1).expand(-1, -1, 3)
    return _on_background(shaded, world_normals, background_rgb)


def shade_matcap(world_normals, camera, background_rgb):
    """View-space normals as RGB, (1 - n_view) / 2, over the background.

    A normal facing the camera has n_view.z < 0 in the OpenCV convention, so
    the negation puts it at the bright end, and a normal facing away is
    flipped toward the camera first.
    """
    H, W, _ = world_normals.shape
    n_view = _view_normals(world_normals, camera)
    n_view = torch.where(n_view[:, 2:3] > 0, -n_view, n_view)
    matcap = ((-n_view + 1.0) * 0.5).reshape(H, W, 3)
    return _on_background(matcap, world_normals, background_rgb)


def compose_full(color, shaded, normals, background_rgb):
    """The Gaussian render at full size beside the shaded and normal panels,
    box-filtered to half size and stacked, on a canvas of H x (W + W // 2)."""
    H, W = color.shape[:2]
    half_h, half_w = H // 2, W // 2
    canvas = np.empty((H, W + half_w, 3), dtype=np.uint8)
    canvas[:] = np.round(np.asarray(background_rgb) * 255.0).astype(np.uint8)
    canvas[:, :W] = color
    for row, panel in enumerate((shaded, normals)):
        canvas[row * half_h:(row + 1) * half_h, W:] = np.asarray(
            Image.fromarray(panel).resize((half_w, half_h), Image.BOX))
    return canvas
