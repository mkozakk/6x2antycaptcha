import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sprite_detector_model.MaskDataset import MaskDataset

cnt = 0
def conditional_display_img(img):
    global cnt
    if cnt > 3:
        return
    cnt += 1
    import matplotlib.pyplot as plt
    plt.imshow(img)





def get_unet(input_shape=(64, 40, 3)):
    inputs = keras.Input(shape=input_shape)
    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    # Bottleneck
    b = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(64, 3, activation='relu', padding='same')(b)
    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(c3)
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(c4)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)
    model = keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    csv_path = os.path.join('data', 'captcha', 'mask_dataset', 'labels.csv')
    images_dir = os.path.join('data', 'captcha', 'mask_dataset', 'images')
    batch_size = 8
    img_size = (64, 40)
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dataset = MaskDataset(train_df, images_dir, batch_size=batch_size, img_size=img_size)
    test_dataset = MaskDataset(test_df, images_dir, batch_size=batch_size, img_size=img_size)
    model = get_unet(input_shape=(img_size[0], img_size[1], 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=10
    )
    model.save('model/sprite_mask_unet.keras')
    print('model saved to model/sprite_mask_unet.keras')

    import matplotlib.pyplot as plt
    model = get_unet(input_shape=(64, 40, 3))
    model.load_weights('model/sprite_mask_unet.keras')
    X, y = test_dataset[0]
    preds = model.predict(X)
    for i in range(len(X)):
        conditional_display_img(X[i])
        conditional_display_img(y[i].squeeze())
        conditional_display_img(preds[i].squeeze())
        plt.show()
