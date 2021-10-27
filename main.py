import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# Setup mixed precision
mixed_precision.set_global_policy('mixed_float16')


# Define residual blocks
def Conv2D_BN_Activation(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu')(x)
        return x

    return block


def Conv2D_Activation_BN(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = Activation('relu')(x)
        x = BatchNormalization(name=bn_name)(x)
        return x

    return block


# Define Residual Block
def Residual_Block(filters, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    def block(input_x):
        x = Conv2D_BN_Activation(filters=filters, kernel_size=(3, 3), padding='same')(input_x)
        x = Conv2D_BN_Activation(filters=filters, kernel_size=(3, 3), padding='same')(x)
        # need convolution on shortcut for add different channel
        if with_conv_shortcut:
            shortcut = Conv2D_BN_Activation(filters=filters, strides=strides, kernel_size=kernel_size)(input_x)
            x = Add()([x, shortcut])
        else:
            x = Add()([x, input_x])
        return x

    return block


# Define tensorboard callbacks
class MinimumEpochEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False, min_epoch=30):
        super(MinimumEpochEarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


class MinimumEpochReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0, verbose=0, mode='auto', factor=0.1, cooldown=0, min_lr=0., min_epoch=30):
        super(MinimumEpochReduceLROnPlateau, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


def train(version_num, batch_size=64):
    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    training_dataset_csv = f"Training Label/public_training_data.csv"
    training_dataset_dir = f"public_training_data/public_training_data/public_training_data"
    checkpoint_path = f'checkpoint_{version_num}.hdf5'
    log_dir = f'logs/{version_num}'
    epochs = 1000
    img_width = 1232
    img_height = 1028
    alphabet = list('ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789 ')
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv(f'{training_dataset_csv}', delimiter=',')
    df['filename'] = df['filename'] + '.jpg'
    df['label'] = df['label'].apply(lambda el: list(el.ljust(12)))
    df[[f'label{i}' for i in range(1, 13)]] = pd.DataFrame(df['label'].to_list(), index=df.index)
    for i in range(1, 13):
        df[f'label{i}'] = df[f'label{i}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=df, directory=training_dataset_dir, subset='training',
                                                  x_col="filename", y_col=[f'label{i}' for i in range(1, 13)],
                                                  class_mode="multi_output",
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    valid_generator = datagen.flow_from_dataframe(dataframe=df, directory=training_dataset_dir, subset='validation',
                                                  x_col="filename", y_col=[f'label{i}' for i in range(1, 13)],
                                                  class_mode="multi_output",
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    input_shape = (img_height, img_width, 3)
    main_input = Input(shape=input_shape)
    x = main_input
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Residual_Block(filters=128, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Residual_Block(filters=256, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D_BN_Activation(filters=256, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Dropout(0.3)(x)
    # x = Conv2D_BN_Activation(filters=512, kernel_size=(3, 3), padding='same')(x)
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # x = Residual_Block(filters=512, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # x = Dropout(0.3)(x)
    # x = Residual_Block(filters=1024, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    # x = main_input
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # x = Dropout(0.2)(x)
    # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # x = Dropout(0.2)(x)
    # x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # # x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
    # # x = BatchNormalization()(x)
    # x = Flatten()(x)
    # x = Dropout(0.4)(x)
    out = [Dense(len(alphabet), name=f'digit{i + 1}', activation='softmax')(x) for i in range(12)]
    model = Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')

    earlystop = MinimumEpochEarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_epoch=20)
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduceLR = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1, mode='auto',
                                             min_lr=0.00001, min_epoch=15)
    callbacks_list = [tensorBoard, earlystop, checkpoint, reduceLR]

    model.summary()
    train_history = model.fit(
        train_generator,
        steps_per_epoch=np.ceil(train_generator.n // train_generator.batch_size),
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=np.ceil(valid_generator.n // valid_generator.batch_size),
        verbose=1,
        callbacks=callbacks_list
    )
    # with open(f"{version_num}.txt", "w") as file:
    #     loss_idx = np.argmin(train_history.history['val_loss'])
    #     digit6_idx = np.argmax(train_history.history['val_digit6_accuracy'])
    #     file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
    #     file.write(f"{train_history.history['val_digit1_accuracy'][loss_idx]}\n")
    #     file.write(f"{train_history.history['val_digit2_accuracy'][loss_idx]}\n")
    #     file.write(f"{train_history.history['val_digit3_accuracy'][loss_idx]}\n")
    #     file.write(f"{train_history.history['val_digit4_accuracy'][loss_idx]}\n")
    #     file.write(f"{train_history.history['val_digit5_accuracy'][loss_idx]}\n")
    #     file.write(f"{train_history.history['val_digit6_accuracy'][loss_idx]}\n")
    #     file.write(f"{'-'*20}\n")
    #     file.write(f"{train_history.history['val_loss'][digit6_idx]}\n")
    #     file.write(f"{train_history.history['val_digit1_accuracy'][digit6_idx]}\n")
    #     file.write(f"{train_history.history['val_digit2_accuracy'][digit6_idx]}\n")
    #     file.write(f"{train_history.history['val_digit3_accuracy'][digit6_idx]}\n")
    #     file.write(f"{train_history.history['val_digit4_accuracy'][digit6_idx]}\n")
    #     file.write(f"{train_history.history['val_digit5_accuracy'][digit6_idx]}\n")
    #     file.write(f"{train_history.history['val_digit6_accuracy'][digit6_idx]}\n")
    K.clear_session()


def main():
    for i in range(2, 10):
        train(version_num=i, batch_size=32)


if __name__ == "__main__":
    main()
