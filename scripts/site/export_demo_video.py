"""Export the recorded global RGB stream of one HDF5 demonstration.

This is a demonstration replay, not a policy evaluation or a new simulation.
"""
import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--trajectory', default='trajectory_000000')
parser.add_argument('--ffmpeg', default='ffmpeg')
args = parser.parse_args()
with h5py.File(args.dataset, 'r') as handle:
    trajectory = handle[args.trajectory]
    frames = trajectory['obs/sensor_data/agentview/rgb']
    times = trajectory['timestamps'][:]
    fps = 1/float(np.median(np.diff(times)))
    height, width = frames.shape[1:3]
    with subprocess.Popen([args.ffmpeg, '-hide_banner', '-loglevel', 'error',
                           '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                           '-framerate', str(fps), '-i', 'pipe:0', '-an', '-map_metadata', '-1',
                           '-c:v', 'libx264', '-threads', '2', '-crf', '22', '-preset', 'fast',
                           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', str(args.output)], stdin=subprocess.PIPE) as process:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        if process.wait(): raise RuntimeError('Video export failed')
    print(f'{args.output.name}: {len(frames)} frames, {fps:.2f} fps')
