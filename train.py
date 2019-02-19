import matplotlib
# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.layers import Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, rmsprop

import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from data_generator import SignDataLoader


def plot_history(history, base_name=""):
    plt.clf()
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(base_name + "accuracy.png")
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(base_name + "loss.png")
    plt.clf()


if __name__ == '__main__':
    batch_size = 1024
    out_classes = ["W11-2", "W11-8", "W1-1_L", "W1-1_R", "W1-2_L", "W1-2_R", "W1-3_L", "W1-3_R", "W1-4_L", "W1-4_R",
                   "W1-5_L", "W1-5_R", "W2-1", "W2-2_L", "W2-2_R", "W3-1", "W3-3", "W4-1_L", "W4-1_R", "W4-2", "W5-2",
                   "W6-2", "W6-3", "W7-1", "W12-1", "W14-1", "W14-2"]  # removed from training: "W1-1a_15_L"
    h_symmetry_classes = [("W1-1_L", "W1-1_R"), ("W1-2_L", "W1-2_R"), ("W1-3_L", "W1-3_R"), ("W1-4_L", "W1-4_R"),
                          ("W1-5_L", "W1-5_R"), ("W2-2_L", "W2-2_R"), ("W4-1_L", "W4-1_R"), ("W1-10_R", "W1-10_L")]
    rotation_and_flips = {"W12-1": ('h',),
                          "W2-1": ('v', 'h', 'd'),
                          "W2-2_L": ('v',),
                          "W2-2_R": ('v',),
                          "W3-1": ('h',),
                          "W3-3": ('h',),
                          "W6-3": ('h',),
                          }
    mapping = {c: i for i, c in enumerate(out_classes)}

    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(96, 96, 3),
                             pooling='avg')

    # MobileNetV2(weights='imagenet', include_top=True).summary()

    predictions = Dense(len(out_classes), activation='softmax')(base_model.outputs[0])
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.summary()
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)
    # exit(0)

    data_loader = SignDataLoader(path_images_dir="/home/nicolas/data/curve",
                                 classes_to_detect=out_classes,
                                 images_size=model.input_shape[1:3],
                                 mapping=mapping,
                                 classes_flip_and_rotation=rotation_and_flips,
                                 symmetric_classes=h_symmetry_classes,
                                 train_test_split=0.2)

    if os.path.isfile("data.npz"):
        savez = np.load("data.npz")
        x_train = savez["x_train"]
        y_train = savez["y_train"]
        x_test = savez["x_test"]
        y_test = savez["y_test"]
        out_classes = savez["out_classes"]
    else:
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()
        y_train = to_categorical(y_train, len(out_classes))
        y_test = to_categorical(y_test, len(out_classes))
        x_train = np.stack([preprocess_input(x) for x in x_train])
        x_test = np.stack([preprocess_input(x) for x in x_test])
        np.savez_compressed("data.npz", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                            out_classes=out_classes)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 brightness_range=(0.3, 1.4),
                                 shear_range=5.0,
                                 zoom_range=(0.7, 1.1),
                                 fill_mode='nearest',
                                 horizontal_flip=False,
                                 vertical_flip=False)

    datagen.fit(x_train)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=rmsprop(lr=0.001, decay=1e-5), loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=ceil(len(x_train) / batch_size),
                                  epochs=60,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  use_multiprocessing=True)
    plot_history(history, "dense_")

    # unfroze the 3 last blocks of mobile net
    for layer in model.layers[:113]:
        layer.trainable = False
    for layer in model.layers[113:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0005, momentum=0.9, decay=1e-6),
                  loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=ceil(len(x_train) / batch_size),
                                  epochs=160,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  use_multiprocessing=True)
    plot_history(history, "fine_tuning_1_")

    model.save("mobilenet_curve_1.h5", overwrite=True)

    # unfroze the 6 last blocks of mobile net
    for layer in model.layers[:87]:
        layer.trainable = False
    for layer in model.layers[87:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0005, momentum=0.9, decay=1e-6),
                  loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=ceil(len(x_train) / batch_size),
                                  epochs=160,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  use_multiprocessing=True)
    plot_history(history, "fine_tuning_2_")

    model.save("mobilenet_curve_2.h5", overwrite=True)

    # # unfroze all mobile net
    # for layer in model.layers:
    #     layer.trainable = True
    #
    # model.compile(optimizer=SGD(lr=0.00001, momentum=0.9, decay=1e-7),
    #               loss='categorical_crossentropy', metrics=["accuracy"])
    # history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    #                               steps_per_epoch=ceil(len(x_train) / batch_size),
    #                               epochs=40,
    #                               verbose=1,
    #                               validation_data=(x_test, y_test),
    #                               use_multiprocessing=True)
    # plot_history(history, "fine_tuning_f_")
    #
    # model.save("mobilenet_curve_f.h5", overwrite=True)





