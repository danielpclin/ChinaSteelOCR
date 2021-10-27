import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import mixed_precision
from functools import reduce

# Setup mixed precision
mixed_precision.set_global_policy('mixed_float16')


def predict(versions=(1,), batch_size=64, method="occur_sum_max"):
    evaluate_dataset_csv = f"submission_template.csv"
    evaluate_dataset_dir = f"public_testing_data/public_testing_data"
    img_width = 1232
    img_height = 1028
    alphabet = list('ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789 ')
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv(f'{evaluate_dataset_csv}', delimiter=',')
    df['id'] = df['id'] + '.jpg'
    predictions = []
    for version in versions:
        checkpoint_path = f'checkpoint_{version}.hdf5'
        image_data_generator = ImageDataGenerator(rescale=1. / 255)
        predict_generator = image_data_generator.flow_from_dataframe(dataframe=df, directory=evaluate_dataset_dir,
                                                                     x_col="id", class_mode=None, shuffle=False,
                                                                     target_size=(img_height, img_width),
                                                                     batch_size=batch_size)
        model = models.load_model(checkpoint_path)
        _prediction = model.predict(
            predict_generator,
            steps=np.ceil(predict_generator.n / predict_generator.batch_size),
            verbose=1,
        )
        predictions.append(_prediction)
        K.clear_session()

    if len(versions) == 1:
        prediction = predictions[0]
        result = ["" for _ in range(len(prediction[0]))]
        prediction = np.argmax(prediction, axis=2)
        for letter in prediction:
            for index, label in enumerate(letter):
                result[index] = result[index] + int_to_char[label % len(alphabet)]
    else:
        prediction_sum_argmax = np.argmax(reduce(np.add, predictions), axis=2)
        prediction_argmax_concat = np.concatenate(np.expand_dims(np.argmax(predictions, axis=3), axis=3), axis=2)
        prediction_concat_argmax = np.argmax(np.concatenate(predictions, axis=2), axis=2)
        result = ["" for _ in range(len(prediction_argmax_concat[0]))]
        if method == "occur_max":
            for letter_index, letter in enumerate(prediction_argmax_concat):
                for label_index, label in enumerate(letter):
                    (values, counts) = np.unique(label, return_counts=True)
                    label = values[counts == counts.max()]
                    if len(label) > 1:
                        result[label_index] = result[label_index] + int_to_char[
                            prediction_concat_argmax[letter_index][label_index] % len(alphabet)]
                    else:
                        result[label_index] = result[label_index] + int_to_char[label[0] % len(alphabet)]
        elif method == "occur_sum_max":
            for letter_index, letter in enumerate(prediction_argmax_concat):
                for label_index, label in enumerate(letter):
                    (values, counts) = np.unique(label, return_counts=True)
                    label = values[counts == counts.max()]
                    if len(label) > 1:
                        result[label_index] = result[label_index] + int_to_char[
                            prediction_sum_argmax[letter_index][label_index] % len(alphabet)]
                    else:
                        result[label_index] = result[label_index] + int_to_char[label[0] % len(alphabet)]
        elif method == "max":
            for letter in prediction_concat_argmax:
                for index, label in enumerate(letter):
                    result[index] = result[index] + int_to_char[label % len(alphabet)]
        elif method == "sum_max":
            for letter in prediction_sum_argmax:
                for index, label in enumerate(letter):
                    result[index] = result[index] + int_to_char[label % len(alphabet)]
    result_df = pd.read_csv(f'{evaluate_dataset_csv}', delimiter=',')
    result_df['text'] = result
    result_df['text'] = result_df['text'].str.strip()
    if len(versions) == 1:
        result_df.to_csv(f'predict/{versions[0]}.csv', index=False)
    else:
        result_df.to_csv(f'predict/{"_".join(map(str, versions))}_{method}.csv', index=False)


if __name__ == "__main__":
    predict(versions=(2,), batch_size=64)
