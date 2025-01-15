import tensorflow as tf
import cv2

def segmentate_captcha(captcha_img):
    h, w = captcha_img.shape[:2]
    cell_w = w // 6
    cell_h = h // 2

    segments = []
    for row in range(2):
        for col in range(6):
            x0 = col * cell_w
            y0 = row * cell_h
            segment = captcha_img[y0:y0+cell_h, x0:x0+cell_w]
            segments.append(segment)

    pairs = {
        'A': (segments[0], segments[3]),   # A: 1 i 7
        'B': (segments[1], segments[4]),   # B: 2 i 8
        'C': (segments[2], segments[5]),   # C: 3 i 9
        'D': (segments[6], segments[9]),   # D: 4 i 10
        'E': (segments[7], segments[10]),  # E: 5 i 11
        'F': (segments[8], segments[11]),  # F: 6 i 12
    }
    return pairs

def solve_captcha(captcha_img):
    pairs = segmentate_captcha(captcha_img)
    results = []

    for k, v in pairs.items():
        img1, img2 = v
        identical = compare_two(img1, img2)
        if identical:
            results.append(k)

    return results

def preprocess_image(img, img_size=(64, 40)):
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    return img

if __name__ == "__main__":
    captcha_img = cv2.imread('../data/captcha/game_probes/s10.jpg')

    results = solve_captcha(captcha_img)
    print("Identical pairs:", results)