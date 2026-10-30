"""Save one frame of a GIF as PNG for visual checking: python scripts/gif_frame.py in.gif out.png frame_index"""
import sys

from PIL import Image

im = Image.open(sys.argv[1])
n = im.n_frames
k = min(int(sys.argv[3]), n - 1)
im.seek(k)
im.convert("RGB").save(sys.argv[2])
print(f"frames={n}, saved frame {k}")
