import sys
import os
import random
import pytest
from PIL import Image, ImageDraw
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generator.data_generator import pick_outfit_frame, insert_letter, add_noise


@pytest.mark.visual
def test_pick_outfit_frame_visual():
    # Losowy plik z katalogu /images/outfits/images
    data_dir = os.path.join('images', 'outfits', 'images')
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    assert files, 'Brak plików w images/outfits/images'
    img_path = os.path.join(data_dir, random.choice(files))
    img = Image.open(img_path)
    cropped = pick_outfit_frame(img, frame=(random.randint(0, 3), random.randint(0, 3)))
    os.makedirs('tests/results', exist_ok=True)
    cropped.save('tests/results/pick_outfit_frame_result.png')
    print('Sprawdź plik tests/results/pick_outfit_frame_result.png')

@pytest.mark.visual
def test_insert_letter_visual():
    img = Image.new('RGB', (43, 64), color='green')
    insert_letter(img, random.choice(list('ABCDEF')), 0, 0)
    os.makedirs('tests/results', exist_ok=True)
    img.save('tests/results/insert_letter_result.png')
    print('Sprawdź plik tests/results/insert_letter_result.png')

@pytest.mark.visual
def test_add_noise_visual():
    img = Image.new("RGB", (256, 128), (128, 128, 128))
    add_noise(img)
    img.save("tests/results/add_noise_result.png")
