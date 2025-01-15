import os
from PIL import Image
import uuid

RAW_DIR = os.path.join(os.path.dirname(__file__), '../images/backgrounds/raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../images/backgrounds/images')
TARGET_SIZE = (256, 128)

os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(RAW_DIR):
    if fname.lower().endswith('.png'):
        fpath = os.path.join(RAW_DIR, fname)
        with Image.open(fpath) as img:
            width, height = img.size
            left = (width - TARGET_SIZE[0]) // 2
            top = (height - TARGET_SIZE[1]) // 2
            right = left + TARGET_SIZE[0]
            bottom = top + TARGET_SIZE[1]
            cropped = img.crop((left, top, right, bottom))
            cropped = cropped.convert('RGB')
            out_name = f"bg_{uuid.uuid4().hex[:8]}.jpg"
            out_path = os.path.join(OUT_DIR, out_name)
            cropped.save(out_path, 'JPEG')

