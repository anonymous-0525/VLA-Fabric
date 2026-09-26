"""Create VP9 alternatives for browsers without an H.264 decoder."""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--ffmpeg', default='ffmpeg')
args = parser.parse_args()

def convert(source):
    output = source.with_suffix('.webm')
    if output.exists() and output.stat().st_mtime > source.stat().st_mtime:
        return
    subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-threads', '2', '-i', str(source), '-an', '-map_metadata', '-1',
                    '-c:v', 'libvpx-vp9', '-threads', '2', '-row-mt', '1',
                    '-deadline', 'realtime', '-cpu-used', '6', '-crf', '36', '-b:v', '0',
                    '-pix_fmt', 'yuv420p', '-y', str(output)], check=True)
    print(output.name, flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(convert, sorted(args.directory.glob('*.mp4'))))
