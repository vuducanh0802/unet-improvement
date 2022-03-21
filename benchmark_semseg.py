import segmentation_models as sm
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
# !pip install git+https://github.com/qubvel/segmentation_models

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description="Testing Benchmarking")
parser.add_argument("--model", type=str, metavar='', required=True, help='name of the model')
parser.add_argument("--dataset", type=dir_path, metavar='', required=True, help='path of the dataset folder')
args = parser.parse_args()
def train(backbone, dataset):
    X = []
    Y = []
    sz = 480
    for jpg in os.listdir(f'{dataset}/images'):
        x = Image.open(f"{dataset}/images/{jpg}")
        y = Image.open(f"{dataset}/result/{jpg}")

        processed_x = np.round(np.array(x.convert('RGB').resize((sz, sz)), dtype=np.float32))
        tf.reshape(processed_x, shape=[-1, sz, sz, 3])
        x = processed_x / 255

        processed_y = np.round(np.array(y.convert('L').resize((sz, sz)), dtype=np.float32))
        tf.reshape(processed_y, shape=[-1, sz, sz, 1])
        y = processed_y / 255

        X.append(x)
        Y.append(y)

    X = np.asarray(X)
    Y = np.asarray(Y)

    x_train, x_val, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    sm.set_framework('tf.keras')

    sm.framework()

    BACKBONE = backbone
    preprocess_input = sm.get_preprocessing(BACKBONE)


    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)


    model = sm.Unet(BACKBONE, encoder_weights='imagenet')

    model.compile(
        'Adam',
        loss=sm.losses.bce_dice_loss,
        metrics=[sm.metrics.accuracy, sm.metrics.iou_score, sm.metrics.precision],
    )

    model.fit(
       x=x_train,
       y=y_train,
       batch_size=4,
       epochs=500,
       validation_data=(x_val, y_test),
    )


if __name__ == '__main__':
    train(args.model, args.dataset)