"""Split approved camera mosaics into synchronized, privacy-filtered web assets.

Input directory contains frame4/relay4/basket3/relay3_brightened.mp4.
Only derived media are written; original recordings are retained unchanged.
"""

import argparse
import json
import subprocess
from pathlib import Path


def run(ffmpeg, source, output, filters, seconds=None):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2",
           "-i", str(source), "-an", "-map_metadata", "-1", "-vf", filters]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(output)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task, arms in [("frame4", 4), ("basket3", 3), ("relay3", 3), ("relay4", 4)]:
        source = args.source/f"{task}_brightened.mp4"
        locations = [(640, 0), (0, 0), (1280, 0), (0, 480), (1280, 480)] if arms == 4 else [(0, 0), (640, 0), (0, 480), (640, 480)]
        for camera, (x, y) in enumerate(locations):
            name = "global" if camera == 0 else f"wrist{camera}"
            output = args.destination/f"{task}-{name}.mp4"
            # Remove the source mosaic's title strip, preserving the full camera width.
            crop = f"crop=640:448:{x}:{y+32},setsar=1"
            if camera:
                # A soft central working region with permanently blurred surroundings.
                # The inner image is mildly filtered too; no unfiltered wrist copy ships.
                privacy = (",split=3[sharp][blur][mask];[blur]gblur=sigma=22[base];"
                           "[sharp]gblur=sigma=2[focus];[mask]format=gray,"
                           "geq=lum='255*clip(min(min((X-80)/50,(560-X)/50),"
                           "min((Y-100)/60,(448-Y)/40)),0,1)'[alpha];"
                           "[base][focus][alpha]maskedmerge")
                crop += privacy
            run(args.ffmpeg, source, output, crop)
            poster = output.with_suffix(".jpg")
            subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "2",
                            "-i", str(output), "-frames:v", "1", "-map_metadata", "-1",
                            "-y", str(poster)], check=True)
            manifest.append({"task": task, "camera": name, "file": output.name,
                             "privacy_filter": "peripheral Gaussian 22px; center 2px" if camera else "none",
                             "audio": False, "timing": "unaltered source timestamps"})
            print(output.name, flush=True)
        run(args.ffmpeg, args.destination/f"{task}-global.mp4",
            args.destination/f"{task}-ambient.mp4", "scale=480:-2,fps=15", seconds=16)
    (args.destination/"media-processing.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
