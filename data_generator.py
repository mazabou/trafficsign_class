import os
from keras.preprocessing.image import load_img, img_to_array
from glob import iglob
import numpy as np
from math import floor
from random import shuffle


class SignDataLoader:
    def __init__(self, path_images_dir, classes_to_detect, images_size, mapping, classes_flip_and_rotation=None,
                 symmetric_classes=None, train_test_split: float = 0.0, classes_merge=None):
        self.base_dir = path_images_dir
        self.classes = classes_to_detect
        self.images_size = images_size
        self.mapping_dict = mapping
        if classes_flip_and_rotation is not None:
            self.classes_flip_and_rotation = classes_flip_and_rotation
        else:
            self.classes_flip_and_rotation = {}
        self.symmetric_classes = {}
        if symmetric_classes is not None:
            for c1, c2 in symmetric_classes:
                self.symmetric_classes[c1] = c2
                self.symmetric_classes[c2] = c1
        self.per_classes_data = {}
        self.train_test_split = train_test_split
        self.classes_merge = classes_merge

    def load_data(self):
        for sign_class in self.classes:
            try:
                sub_class_list = self.classes_merge[sign_class]
                if sign_class not in sub_class_list:
                    sub_class_list.append(sign_class)
            except (KeyError, TypeError):
                sub_class_list = [sign_class]
            for sub_class in sub_class_list:
                for image_path in iglob(os.path.join(self.base_dir, sub_class, "*.jpg")):
                    try:
                        img = load_img(image_path, target_size=self.images_size)
                    except IOError:
                        print("Unable to read file {}".format(image_path))
                        continue
                    img = img_to_array(img)
                    if sign_class in self.classes_flip_and_rotation:
                        for transformed_image in self.apply_transform(img, self.classes_flip_and_rotation[sign_class]):
                            self.add_to_train_data(transformed_image, sign_class)
                    else:
                        self.add_to_train_data(img, sign_class)
        x_train, x_test = [], []
        y_train, y_test = [], []
        for sign_class, sign_images in self.per_classes_data.items():
            test_count = floor(len(sign_images) * self.train_test_split)
            shuffle(sign_images)
            x_test += sign_images[:test_count]
            x_train += sign_images[test_count:]
            class_id = self.mapping_dict[sign_class]
            y_test += [class_id] * test_count
            y_train += [class_id] * (len(sign_images) - test_count)
        for x in x_train + x_test:
            assert x.shape == (self.images_size[0], self.images_size[1], 3)
        return (np.stack(x_train), np.array(y_train)), (np.stack(x_test), np.array(y_test))

    def apply_transform(self, img: np.ndarray, transform_list):
        if len(transform_list) == 0:
            return [img]
        current_transform = transform_list[0]
        image_list = self.apply_transform(img, transform_list[1:])
        if current_transform == 'd':
            return image_list + self.apply_transform(img.transpose([1, 0, 2]), transform_list[1:])
        elif current_transform == 'v':
            return image_list + self.apply_transform(np.flipud(img), transform_list[1:])
        elif current_transform == 'h':
            return image_list + self.apply_transform(np.fliplr(img), transform_list[1:])

    def add_to_train_data(self, img: np.ndarray, label, check_symmetry=True):
        if not check_symmetry or label not in self.symmetric_classes:
            try:
                self.per_classes_data[label].append(img)
            except KeyError:
                self.per_classes_data[label] = []
                self.per_classes_data[label].append(img)
        else:
            sym = self.symmetric_classes[label]
            self.add_to_train_data(img, label, check_symmetry=False)
            self.add_to_train_data(np.fliplr(img), sym, check_symmetry=False)




