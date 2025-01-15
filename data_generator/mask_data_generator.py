import os
import random
import csv
import math
import colorsys
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import multiprocessing
import threading
from queue import Queue

class SampleSaver:
    def __init__(self, labels_path):
        self.labels_path = labels_path
        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        with open(self.labels_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'file', 'mask'])

    def save(self, samples):
        with open(self.labels_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(samples)

class AsyncImageSaver:
    def __init__(self, out_dir, num_workers=8, total_images=None):
        self.out_dir = out_dir
        self.queue = Queue()
        self.threads = []
        self.stop_signal = object()
        self.total_images = total_images
        self.progress = tqdm(total=total_images, desc='Saving images') if total_images else None
        for _ in range(num_workers):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def worker(self):
        while True:
            item = self.queue.get()
            try:
                if item is self.stop_signal:
                    break
                img, filename = item
                img.save(os.path.join(self.out_dir, filename), format='JPEG', quality=80, subsampling=2, optimize=True)
                if self.progress:
                    self.progress.update(1)
            finally:
                self.queue.task_done()

    def save_async(self, img, filename):
        self.queue.put((img, filename))

    def close(self):
        for _ in self.threads:
            self.queue.put(self.stop_signal)
        self.queue.join()
        for t in self.threads:
            t.join()
        if self.progress:
            self.progress.close()

def pick_outfit_frame(img: Image.Image, frame=(0, 0)) -> Image.Image:
    seg_width = img.width // 4
    seg_height = img.height // 4

    row, col = frame

    left = col * seg_width
    upper = row * seg_height
    right = left + seg_width
    lower = upper + seg_height

    return img.crop((left, upper, right, lower))

def insert_letter(img: Image.Image, letter: str, font, x0=0, y0=0):
    draw = ImageDraw.Draw(img)
    offset_x, offset_y = 3, 4
    base_x, base_y = x0 + offset_x, y0 + offset_y
    for dx in range(-2, 2):
        for dy in range(-2, 2):
            if dx != 0 or dy != 0:
                draw.text((base_x + dx, base_y + dy), letter, font=font, fill="white")
    draw.text((base_x, base_y), letter, font=font, fill="black")

def add_noise(img: Image.Image):
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for _ in range(random.randint(3, 5)):
        hue = random.random()
        brightness = random.uniform(0.7, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, brightness)
        color = tuple(int(255 * c) for c in rgb)
        x0 = random.randint(0, width - 1)
        y0 = 0
        x1 = random.randint(0, width - 1)
        y1 = height - 1
        draw.line([(x0, y0), (x1, y1)], fill=color, width=1)

    for _ in range(random.randint(2, 5)):
        hue = random.random()
        brightness = random.uniform(0.7, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, brightness)
        color = tuple(int(255 * c) for c in rgb)
        size = random.randint(min(width, height) // 10, min(width, height) // 5)
        angle = random.uniform(0, 2 * math.pi)
        cx = random.randint(size, width - size)
        cy = random.randint(size, height - size)
        points = []
        for i in range(3):
            theta = angle + i * 2 * math.pi / 3
            x = cx + size * math.cos(theta)
            y = cy + size * math.sin(theta)
            points.append((x, y))
        draw.polygon(points, outline=color, width=1)

def load_images_from_dir(directory, mode='RGB'):
    images = {}
    for fname in os.listdir(directory):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(directory, fname)
            img = Image.open(path).convert(mode)
            images[path] = img
    return images

def worker_generate_sample(args):
    (bg_img, outfit_images, outfit_files, segment_pairs, letters, rows, cols, sample_idx, id_start, font, img_saver) = args
    samples = []
    segments = []
    bg = bg_img.copy()
    add_noise(bg)
    width, height = bg.size
    seg_w, seg_h = width // cols, height // rows
    for r in range(rows):
        for c in range(cols):
            left = c * seg_w
            upper = r * seg_h
            right = left + seg_w
            lower = upper + seg_h
            seg = bg.crop((left, upper, right, lower))
            segments.append(seg)
    local_id = id_start
    for pair_idx, ((i, j), letter) in enumerate(zip(segment_pairs, letters)):
        frame = (random.randint(0, 3), random.randint(0, 3))
        if random.random() < 0.5:
            outfit_path = random.choice(outfit_files)
            is_same = 1
            frame2 = frame
            outfit_path2 = outfit_path
        else:
            outfit_path = random.choice(outfit_files)
            outfit_path2 = random.choice([f for f in outfit_files if f != outfit_path])
            is_same = 0
            frame2 = frame
        insert_letter(segments[i], letter, font)
        insert_letter(segments[j], letter, font)
        outfit_img = outfit_images[outfit_path]
        out1 = pick_outfit_frame(outfit_img, frame)
        offset_x1 = random.randint(1, 10)
        offset_y1 = random.randint(2, 12)
        segments[i].paste(out1, (offset_x1, offset_y1), out1 if out1.mode == 'RGBA' else None)
        file1 = f'm_{local_id}.jpg'
        file2 = f'm_{local_id}_mask.jpg'
        img_saver.save_async(segments[i].copy(), file1)
        # Generowanie maski
        mask = Image.new('L', (seg_w, seg_h), 0)
        if out1.mode == 'RGBA':
            alpha = out1.split()[-1]
            # Zamiana na maskę binarną: 0 lub 255
            binary_alpha = alpha.point(lambda p: 255 if p > 0 else 0)
            mask.paste(binary_alpha, (offset_x1, offset_y1))
        else:
            mask.paste(255, (offset_x1, offset_y1))
        img_saver.save_async(mask, file2)
        samples.append([local_id, file1, file2])
        local_id += 1
    return samples

def generate_sample():
    bg_dir = os.path.join('data', 'backgrounds', 'data')
    outfit_dir = os.path.join('data', 'outfits', 'data')
    out_dir = os.path.join('data', 'captcha', 'mask_dataset', 'images')
    labels_path = os.path.join('data', 'captcha', 'mask_dataset', 'labels.csv')
    bg_images = load_images_from_dir(bg_dir, mode='RGB')
    outfit_images = load_images_from_dir(outfit_dir, mode='RGBA')
    bg_files = list(bg_images.keys())
    outfit_files = list(outfit_images.keys())
    if not bg_files or not outfit_files:
        print('Brak plików tła lub outfitów!')
        return

    segment_pairs = [(0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)]
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    rows, cols = 2, 6
    SAMPLES_PER_BG = 20
    saver = SampleSaver(labels_path)
    os.makedirs(out_dir, exist_ok=True)
    global_id = 0
    font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 14)
    total_images = len(bg_files) * SAMPLES_PER_BG * len(segment_pairs)
    img_saver = AsyncImageSaver(out_dir, num_workers=4, total_images=total_images)
    with tqdm(total=len(bg_files) * SAMPLES_PER_BG, desc='All samples') as pbar:
        for bg_path in tqdm(bg_files, desc='Backgrounds'):
            bg_img = bg_images[bg_path]
            args_list = []
            for i in range(SAMPLES_PER_BG):
                id_start = global_id + i * len(segment_pairs)
                args_list.append((bg_img, outfit_images, outfit_files, segment_pairs, letters, rows, cols, i, id_start, font, img_saver))
            for args in args_list:
                result = worker_generate_sample(args)
                saver.save(result)
                pbar.update(1)
            global_id += SAMPLES_PER_BG * len(segment_pairs)
    img_saver.close()

if __name__ == '__main__':
    generate_sample()
