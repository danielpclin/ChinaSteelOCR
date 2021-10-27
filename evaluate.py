import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import mixed_precision

# Setup mixed precision
mixed_precision.set_global_policy('mixed_float16')


def evaluate(versions=(1,), batch_size=64, method="ocuur_sum_max"):
    evaluate_dataset_csv = f"Training Label/public_testing_data.csv"
    evaluate_dataset_dir = f"public_training_data/public_training_data/public_testing_data"
    img_width = 1232
    img_height = 1028
    alphabet = list('ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789 ')
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv(f'{evaluate_dataset_csv}', delimiter=',')
    df['filename'] = df['filename'] + '.jpg'
    df['label'] = df['label'].apply(lambda el: list(el.ljust(12)))
    df[[f'label{i}' for i in range(1, 13)]] = pd.DataFrame(df['label'].to_list(), index=df.index)
    for i in range(1, 13):
        df[f'label{i}'] = df[f'label{i}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
    for version in versions:
        checkpoint_path = f'checkpoint_{version}.hdf5'
        image_data_generator = ImageDataGenerator(rescale=1. / 255)
        evaluate_generator = image_data_generator.flow_from_dataframe(dataframe=df, directory=evaluate_dataset_dir,
                                                                      x_col="filename",
                                                                      y_col=[f'label{i}' for i in range(1, 13)],
                                                                      class_mode="multi_output", shuffle=False,
                                                                      target_size=(img_height, img_width),
                                                                      batch_size=batch_size)
        model = models.load_model(checkpoint_path)
        evaluation = model.prediction(
            evaluate_generator,
            steps=np.ceil(evaluate_generator.n / evaluate_generator.batch_size),
            verbose=1,
        )
        print(evaluation)
        K.clear_session()


if __name__ == "__main__":
    evaluate(versions=(2,), batch_size=64)
