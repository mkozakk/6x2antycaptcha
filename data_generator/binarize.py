import os
from PIL import Image

input_dir = 'data/captcha/mask_dataset/images'

for fname in os.listdir(input_dir):
    if 'mask' not in fname.lower():
        continue
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert('L')
    bin_img = img.point(lambda p: 255 if p > 127 else 0)
    bin_img.save(img_path, format='JPEG')
print('done')