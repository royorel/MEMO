"""Video encoding of the rendered frames with ffmpeg."""
import itertools
import subprocess

import numpy as np


# The encoder and its arguments per codec, besides the CRF. The H.265 tag
# lets Safari play the file.
CODECS = {
    "av1": ["-c:v", "libaom-av1", "-b:v", "0", "-cpu-used", "4", "-row-mt", "1"],
    "h264": ["-c:v", "libx264", "-preset", "slow"],
    "h265": ["-c:v", "libx265", "-preset", "slow", "-tag:v", "hvc1"],
}
DEFAULT_CRF = {"av1": 30, "h264": 10, "h265": 10}


def write_video(frames, path, fps, codec, crf):
    """Encode `frames`, HxWx3 uint8 RGB arrays of one size, as the video at
    `path`. 4:2:0 chroma needs even dimensions, so an odd last row or column
    is cropped."""
    frames = iter(frames)
    first = next(frames)
    height, width = first.shape[:2]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-",
        # BT.709 conversion and tags, which players assume for HD video, so
        # no color shift separates the video from its frames.
        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0,scale=out_color_matrix=bt709:out_range=tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        *CODECS[codec], "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in itertools.chain([first], frames):
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
    except BrokenPipeError:
        # ffmpeg stopped reading, and its exit status below says why.
        pass
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
