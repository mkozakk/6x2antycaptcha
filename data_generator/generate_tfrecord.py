import tensorflow as tf
import pandas as pd
import os

csv_path = 'data/captcha/dataset/labels.csv'
images_dir = 'data/captcha/dataset/images'
tfrecord_path = 'data/captcha/dataset/pairs2.tfrecord'

def create_example(image1_bytes, image2_bytes, label):
    feature = {
        'image1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1_bytes])),
        'image2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

df = pd.read_csv(csv_path)

with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for idx, row in df.iterrows():
        path1 = os.path.join(images_dir, row['file1'])
        path2 = os.path.join(images_dir, row['file2'])
        with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
            img1 = f1.read()
            img2 = f2.read()
        label = int(row['is_same_frame'])
        example = create_example(img1, img2, label)
        writer.write(example.SerializeToString())

print(f"Zapisano {len(df)} przykładów do {tfrecord_path}")