import io
import random
import struct
import tempfile
from contextlib import contextmanager
from typing import List, Union

import numpy as np
import PIL.Image
import PIL.ImageOps

from .import_utils import (
    BACKENDS_MAPPING,
    is_opencv_available,
)
from .logging import get_logger
import ffmpeg


global_rng = random.Random()

logger = get_logger(__name__)


@contextmanager
def buffered_writer(raw_f):
    f = io.BufferedWriter(raw_f)
    yield f
    f.flush()


def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None, fps: int = 10) -> str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    image[0].save(
        output_gif_path,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )
    return output_gif_path


def export_to_ply(mesh, output_ply_path: str = None):
    """
    Write a PLY file for a mesh.
    """
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

    coords = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)

    with buffered_writer(open(output_ply_path, "wb")) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))

    return output_ply_path


def export_to_obj(mesh, output_obj_path: str = None):
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)
    vertices = [
        "{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())
    ]

    faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]

    combined_data = ["v " + vertex for vertex in vertices] + faces

    with open(output_obj_path, "w") as f:
        f.writelines("\n".join(combined_data))


def export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
) -> str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [((frame/2 +0.5).clip(0, 1) * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)#cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        print('img max is', np.max(img))
        print('img min is', np.min(img))
        video_writer.write(img)
    return output_video_path


# def export_to_video_hdr(
#     video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
# ) -> str:
#     if is_opencv_available():
#         import cv2
#     else:
#         raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
#     if output_video_path is None:
#         output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

#     if isinstance(video_frames[0], np.ndarray):
#         video_frames = [(np.exp(frame)).astype(np.float32) for frame in video_frames]

#     elif isinstance(video_frames[0], PIL.Image.Image):
#         video_frames = [np.array(frame) for frame in video_frames]

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     h, w, c = video_frames[0].shape
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
#     for i in range(len(video_frames)):
#         img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)#cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
#         print('img max is', np.max(img))
#         print('img min is', np.min(img))
#         video_writer.write(img)
#     return output_video_path


def export_to_video_hdr(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], 
    save_folder: str = None,
    output_video_path: str = None, fps: int = 10,
    min_val: float = None, max_val: float = None  # 添加 min_val 和 max_val 参数来指定原始数据范围
) -> str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError("OpenCV is required for export_to_video_hdr")

    # Handle default output path
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    # Convert PIL images to NumPy arrays if needed
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    video_frames = [np.exp(frame).astype(np.float32) for frame in video_frames]
    # 自动计算 min_val 和 max_val，如果没有指定
    if min_val is None or max_val is None:
        all_frames = np.concatenate([frame.flatten() for frame in video_frames])
        min_val = np.min(all_frames) if min_val is None else min_val
        max_val = np.max(all_frames) if max_val is None else max_val
        print(f"Auto-calculated min: {min_val}, max: {max_val}")

    # Ensure the frames are in HDR format (float32 for precision, apply any necessary transforms)
    # video_frames = [(frame.astype(np.float32)) for frame in video_frames]

    # Get frame dimensions
    h, w, c = video_frames[0].shape

    # Prepare ffmpeg input pipe and output
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s=f'{w}x{h}', framerate=fps)
        .output(
            output_video_path,
            pix_fmt='yuv420p10le',  # 10-bit HDR format
            vcodec='libx265',  # Use x265 codec for HDR encoding
            crf=18,  # Constant rate factor for quality control
            color_primaries='bt2020',  # BT.2020 color space
            color_trc='smpte2084',  # PQ transfer function (ST.2084)
            colorspace='bt2020nc',  # Non-constant luminance BT.2020
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    # Process and write each frame to ffmpeg pipe
    for i in range(len(video_frames)):
        img = video_frames[i]

        # 线性映射原始 HDR 数据到 [0, 65535]
        print('img max is', np.max(img))
        print('img min is', np.min(img))
       
        cv2.imwrite(f'{save_folder}/img_{i}.hdr', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_normalized = (img - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
        img_mapped = np.clip(img_normalized * 65535, 0, 65535).astype(np.uint16)  # 映射到 [0, 65535]
        
        # Convert to RGB format for ffmpeg (OpenCV uses BGR, but ffmpeg expects RGB)
        img_rgb = cv2.cvtColor(img_mapped, cv2.COLOR_BGR2RGB)

        # Write frame to ffmpeg
        process.stdin.write(img_rgb.tobytes())

    # Close the ffmpeg process
    process.stdin.close()
    process.wait()

    return output_video_path