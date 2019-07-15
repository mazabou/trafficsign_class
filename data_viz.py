import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input

from data_generator import SignDataLoader

from train import classes, get_data_for_master_class


if __name__ == '__main__':
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

    data_loader = SignDataLoader(path_images_dir="/home/nicolas/data/curve",
                                 classes_to_detect=out_classes,
                                 images_size=(96, 96),
                                 mapping=mapping,
                                 classes_flip_and_rotation=rotation_and_flips,
                                 symmetric_classes=h_symmetry_classes,
                                 train_test_split=0.2)

    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # x_train = np.stack([preprocess_input(x) for x in x_train])
    # x_test = np.stack([preprocess_input(x) for x in x_test])

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

    # for im in x_train:
    #     print(im.shape, im.min(), im.max(), im.mean())
    #     plt.imshow(im.astype(np.int))
    #     plt.show()

    for b in datagen.flow(x_train, y_train, batch_size=1):
        im, im_class = b[0][0], b[1][0]
        print(im.shape, im.min(), im.max(), im.mean())
        print(im_class, out_classes[im_class])
        plt.imshow(im.astype(np.int))
        plt.show()


