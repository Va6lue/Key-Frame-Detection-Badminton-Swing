import keras
from keras.api.utils import to_categorical
from keras.api.layers import Input, Dense, Dropout
from keras.api.regularizers import l2
from keras.api.callbacks import History, EarlyStopping
from keras.api.models import load_model

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from dataset import load_data, split_data, oversample, load_data_trial
from utils import plot_test_auroc, plot_test_confusion_matrix, get_keyframe_positions, compute_avg_position_error


def model_mlp(in_features, num_classes):
    model = keras.Sequential()
    model.add(Input((in_features,)))
    model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.0001)))
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.0001)))
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def plot_train_result(history: History, save_name=None):
    # acc
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    if save_name is not None:
        plt.savefig(f'my_code/results/{save_name}_train_acc.jpg')
    else:
        plt.show()

    plt.clf()  # clear the current figure

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    if save_name is not None:
        plt.savefig(f'my_code/results/{save_name}_train_loss.jpg')
    else:
        plt.show()


def keras_train(
    data: tuple[np.ndarray],
    model_name: str,
    save_name=None,
    model_path=None
):
    epochs = 500
    batch_size = 32
    lr = 1e-5
    earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')

    X_train, y_train, X_val, y_val = data

    match model_name:
        case 'mlp':
            model = model_mlp(X_train.shape[1], num_classes)
        case _:
            raise NotImplementedError

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            keras.metrics.F1Score('macro', threshold=0.5),
        ]
    )
    model.summary()

    history: History = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[earlystop]
    )

    if save_name is not None:
        model.save(model_path)
        plot_train_result(history, save_name)
    else:
        plot_train_result(history)

    return model


if __name__ == '__main__':
    num_classes = 4

    data_x, data_y = load_data(
        x_path='all_coordinates_list.csv',
        y_path='label_all_keyframe_add.csv',
        plot_class_distribution=False
    )

    X_train, y_train, \
    X_val, y_val, \
    X_test, y_test = split_data(data_x, data_y, train_ratio=0.8)

    X_train, y_train = oversample(X_train, y_train)

    # Convert to binary class matrix
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model_name = 'mlp'
    save_name = model_name
    model_path = f'my_code/weights/{save_name}.keras'

    # Get the trained model
    if Path(model_path).exists():
        model: keras.Model = load_model(model_path)
        model.summary()
    else:
        model = keras_train(
            data=(X_train, y_train, X_val, y_val),
            model_name=model_name,
            save_name=save_name,
            model_path=model_path
        )

    # Test the test set
    y_pred = model.predict(X_test)
    plot_test_auroc(y_test, y_pred, model_name, save_name, save=True)
    plot_test_confusion_matrix(y_test, y_pred, model_name, save_name, save=True)

    # Test another test set: 'data_trial'
    data_trial_path = Path('data_trial')
    data, videos_len, true_positions = load_data_trial(data_trial_path)

    n, t, d = data.shape
    y_pred: np.ndarray = model.predict(data.reshape(-1, d))
    y_pred = y_pred.reshape(n, t, -1)

    y_pos = get_keyframe_positions(y_pred, videos_len, no_ordering=True)
    y_pos_ordered = get_keyframe_positions(y_pred, videos_len, no_ordering=False)

    ape_score = compute_avg_position_error(y_pos, true_positions, model_name=model_name)
    ape_score_ordered = compute_avg_position_error(y_pos_ordered, true_positions, model_name=model_name)
    
    print('Test trial =>')
    pd.set_option('display.float_format', '{:.1f}'.format)
    print(ape_score)
    print(ape_score_ordered)
    print()
