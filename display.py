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
import json
from collections import defaultdict
import glob
import cv2

from data_generator import SignDataLoader


def test_and_display(sign_folder_path: str, model_path: str, classes_json_path: str):
    with open(classes_json_path, 'r') as classes_json:
        category_dict = json.load(classes_json)
    d = {v: k for k, v in category_dict.items()}

    img_rows = 96
    img_cols = 96
    num_classes = len(category_dict)

    # ===========================MobileNetV2==========================================
    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(96, 96, 3),
                             pooling='avg')
    predictions = Dense(num_classes, activation='softmax')(base_model.outputs[0])
    model = Model(inputs=base_model.input, outputs=predictions)
    # ================================================================================

    # Training
    print("Loading weights")
    model.load_weights(model_path)
    print("Weights loaded")

    print("finding images")
    images = glob.glob(os.path.join(sign_folder_path, "*", "*.jpg"))
    print("images listed")

    while True:
        image_path = np.random.choice(images, 1)[0]
        print(image_path)

        frame = cv2.imread(image_path)
        resized = cv2.resize(frame, (img_rows, img_cols))
        preprocesed = preprocess_input(resized.astype(np.float32))

        pred = model.predict(preprocesed.reshape((1,) + preprocesed.shape), verbose=0)
        pred = pred[0]

        pred_index = np.argsort(pred)[::-1]
        print(pred_index)

        top = 3

        right_class_found = False
        gt_class = image_path.split('/')[-2]
        for i, idx in enumerate(pred_index[:top]):
            pred_class = d[idx]
            print("{}:\t{}".format(pred_class, pred[idx]), end=" ")
            if gt_class == pred_class:
                print("Right class")
                right_class_found = True
            else:
                print()
        if not right_class_found:
            if gt_class in category_dict:
                print("The right class was {}, confidence for this class was {}, rank: {}, id: {}"
                      .format(gt_class, pred[category_dict[gt_class]], np.where(pred_index == category_dict[gt_class]),
                              category_dict[gt_class]))
            else:
                print("Model was not trained for class {}".format(gt_class))

        plt.figure(0)
        plt.subplot(1, 3, 1)
        plt.imshow(frame)
        plt.subplot(1, 3, 2)
        plt.imshow(resized)
        plt.subplot(1, 3, 3)
        p = preprocesed - preprocesed.min()
        p = p / p.max()
        plt.imshow(p)
        plt.show()
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("names", type=str)
    parser.add_argument("images", type=str)
    args = parser.parse_args()

    test_and_display(args.images, args.model, args.names)

