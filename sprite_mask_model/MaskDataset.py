from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from PIL import Image


class MaskDataset(Sequence):
    def __init__(self, data, images_dir, batch_size=8, img_size=(64, 40)):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.reset_index(drop=True)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        masks = []
        for _, row in batch.iterrows():
            img_path = os.path.join(self.images_dir, row['file'])
            mask_path = os.path.join(self.images_dir, row['mask'])
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_jpeg(mask, channels=1)
            mask = tf.cast(mask > 50, tf.float32)


            # image = image[:, 1:-1, :]
            # mask = mask[:, 1:-1, :]

            image = tf.image.resize(image, self.img_size)
            mask = tf.image.resize(mask, self.img_size)

            image = tf.cast(image, tf.float32) / 255.0
            mask = tf.cast(mask > 0, tf.float32)
            images.append(image.numpy())
            masks.append(mask.numpy())
        return np.stack(images), np.stack(masks)


