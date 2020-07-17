import matplotlib

# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

from keras.applications import mobilenetv2, inception_resnet_v2, nasnet
from keras.layers import Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, rmsprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import json

from PIL import Image

from data_generator import SignDataLoader


# Each classes must define three keys: signs_classes, h_symmetry and rotation_and_flips
# signs_classes: list of the class to use for training this particular super class.
# h_symmetry: horizontal simetry that transform a sign class into an other one, for example turn left become turn right
# rotation_and_flips: transformations and flips that keep the sign in the same class, 'v' vertical symmetry,
#   'h' horizontal and 'd' transposition.
classes = {
    "Rectangular": {
        "signs_classes": ["W13-1P_10", "W13-1P_15", "W13-1P_20", "W13-1P_25", "W13-1P_30",
                          "W13-1P_35", "W13-1P_45", "W16-7P", "W1-8_L", "W1-8_R", "W1-7",
                          "W1-6_L", "W1-6_R", "rectangle-other"],
        "h_symmetry": [("W1-6_L", "W1-6_R"), ("W1-8_L", "W1-8_R")],
        "rotation_and_flips": {"W1-7": ('h', 'v'),
                               "rectangle-other": ('v', 'h', 'd')}
    },
    "Diamond": {
        "signs_classes": ["W11-2", "W11-8", "W1-1_L", "W1-1_R", "W1-2_L", "W1-2_R", "W1-3_L", "W1-3_R", "W1-4_L",
                          "W1-4_R",
                          "W1-5_L", "W1-5_R", "W2-1", "W2-2_L", "W2-2_R", "W3-1", "W3-3", "W4-1_L", "W4-1_R", "W4-2",
                          "W5-2",
                          "W6-2", "W6-3", "W7-1", "W12-1", "W14-1", "W14-2", "diamond-other", "WorkZone"],
        # removed from training: "W1-1a_15_L"
        "h_symmetry": [("W1-1_L", "W1-1_R"), ("W1-2_L", "W1-2_R"), ("W1-3_L", "W1-3_R"), ("W1-4_L", "W1-4_R"),
                       ("W1-5_L", "W1-5_R"), ("W2-2_L", "W2-2_R"), ("W4-1_L", "W4-1_R"), ("W1-10_R", "W1-10_L")],
        "rotation_and_flips": {"W12-1": ('h',),
                               "W2-1": ('v', 'h', 'd'),
                               "W2-2_L": ('v',),
                               "W2-2_R": ('v',),
                               "W3-1": ('h',),
                               "W3-3": ('h',),
                               "W6-3": ('h',),
                               }
    },
    "Zebra": {
        "signs_classes": ["OM3-L", "OM3-R"],
        "h_symmetry": [("OM3-L", "OM3-R")],
        "rotation_and_flips": {"OM3-L": ('d',), "OM3-R": ('d',)}
    },

    "RedRoundSign": {
        "signs_classes": ['p1', 'p10', 'p11', 'p12', 'p19', 'p20', 'p23', 'p26', 'p27', 'p3', 'p5L', 'p6', 'p9', 'pax',
                          'pb', 'phx', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70',
                          'pl80', 'pmx', 'p_prohibited_bicycle_and_pedestrian', 'p_prohibited_bus_and_truck',
                          'p_prohibited_other', 'prx', 'p_other', 'plo'],
        "merge_sign_classes": {
            "prx": ['pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'prx'],
            "pmx": ['pm1.5', 'pm10', 'pm13', 'pm15', 'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46',
                    'pm5', 'pm49', 'pm50', 'pm55', 'pm8'],
            "phx": ['ph', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.x', 'ph3',
                    'ph3.2', 'ph3.3', 'ph3.5', 'ph3.7', 'ph3.8', 'ph38', 'ph39', 'ph45', 'ph4', 'ph4.2', 'ph4.3',
                    'ph4.4', 'ph4.5', 'ph4.6', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'ph6'],
            "pax": ['pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax'],
            "plo": ['pl35', 'pl25', 'pl15', 'pl10', 'pl110', 'pl65', 'pl90'],
            "p_other": ['pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5',
                        'p_prohibited_two_wheels_vehicules', 'p_prohibited_bicycle_and_pedestria',
                        'p_prohibited_bicycle_and_pedestrian_issues', 'p13', 'p15', 'p16', 'p17', 'p18', 'p2', 'p21',
                        'p22', 'p24', 'p25', 'p28', 'p4', 'p5R', 'p7L', 'p7R', 'p8', 'p15', 'pc']
            },
        "h_symmetry": [],
        "rotation_and_flips": {  # "pne": ('v', 'h', 'd'),
            # "pn": ('v', 'h', 'd'),
            # "pnl": ('d',),
            # "pc": ('v', 'h', 'd'),
            "pb": ('v', 'h', 'd'),
            "p_other": ('v', 'h', 'd'),
        }
    },
}


def plot_history(history, base_name=""):
    plt.clf()
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(base_name + "accuracy.png")
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(base_name + "loss.png")
    plt.clf()


def get_data_for_master_class(class_name: str, mapping, mapping_id_to_name, rotation_and_flips, data_dir: str,
                              merge_sign_classes, h_symmetry_classes, image_size, ignore_npz: bool, out_classes):
    data_file_path = "{0}/{0}.npz".format(class_name)
    if os.path.isfile(data_file_path) and not ignore_npz:
        savez = np.load(data_file_path)
        x_train = savez["x_train"]
        y_train = savez["y_train"]
        x_test = savez["x_test"]
        y_test = savez["y_test"]
    else:
        data_loader = SignDataLoader(path_images_dir=data_dir,
                                     classes_to_detect=out_classes,
                                     images_size=image_size,
                                     mapping=mapping,
                                     classes_flip_and_rotation=rotation_and_flips,
                                     symmetric_classes=h_symmetry_classes,
                                     train_test_split=0.2,
                                     classes_merge=merge_sign_classes)
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()
        with open("{0}/{0}_class_counts.json".format(class_name), 'w') as count_json:
            train_names, train_counts = np.unique(y_train, return_counts=True)
            test_names, test_counts = np.unique(y_test, return_counts=True)
            counts = {n: {"train": 0, "test": 0} for n in mapping.keys()}
            for c, count in zip(train_names, train_counts):
                c_name = mapping_id_to_name[c]
                counts[c_name]["train"] = int(count)
            for c, count in zip(test_names, test_counts):
                c_name = mapping_id_to_name[c]
                counts[c_name]["test"] = int(count)
            json.dump(obj=counts, fp=count_json, indent=4)
        y_train = to_categorical(y_train, len(out_classes))
        y_test = to_categorical(y_test, len(out_classes))
        np.savez_compressed(data_file_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                            out_classes=out_classes)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    with open("{0}/{0}_mapping.json".format(class_name), 'w') as json_mapping:
        json.dump(mapping, json_mapping, indent=4)

    return x_train, y_train, x_test, y_test


def bottleneck_resize(img, bottleneck_min_size=32):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    assert height == width
    bottleneck_size = np.random.randint(bottleneck_min_size, height)
    img_pil = Image.fromarray(np.array(img, dtype=np.uint8))
    down_img = img_pil.resize((bottleneck_size, bottleneck_size), resample=Image.BILINEAR)
    up_img = down_img.resize((height,height), resample=Image.BILINEAR)
    return np.array(up_img)


def random_crop(img, min_reshape=64):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy = np.random.randint(min_reshape, height)
    dx = np.random.randint(min_reshape, width)
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    img_pil = Image.fromarray(np.array(img[y:(y + dy), x:(x + dx), :], dtype=np.uint8))
    up_img = img_pil.resize((height,width), resample=Image.BILINEAR)
    return np.array(up_img)


def custom_transform(batches):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = bottleneck_resize(random_crop(batch_x[i]))
        yield (batch_crops, batch_y)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_class", type=str, choices=classes.keys())
    parser.add_argument("data_dir", type=str)
    parser.add_argument('-e', "--epoch",
                        required=False,
                        type=int,
                        default=64,
                        dest="epoch")
    parser.add_argument('-ef', "--epoch-fine-tune",
                        required=False,
                        type=int,
                        default=200,
                        dest="epoch_fine_tune")
    parser.add_argument('-b', '--batch-size',
                        required=False,
                        default=1024,
                        type=int,
                        dest="batch")
    parser.add_argument('-lr', '--learning-rate',
                        required=False,
                        default=1e-4,
                        type=float,
                        dest="lr")
    parser.add_argument('-decay', '--learning-rate-decay',
                        required=False,
                        default=1e-6,
                        type=float,
                        dest="decay")
    parser.add_argument('-ignore-npz', '--ignore-precomputed-learning-file',
                        required=False,
                        default=False,
                        type=bool,
                        dest="ignore_npz")
    parser.add_argument('-ri', '--use-random-weight-initialisation',
                        required=False,
                        default=False,
                        type=bool,
                        dest="random_init")
    parser.add_argument('-ua', '--unfroze-all-convolution-layer-directly',
                        required=False,
                        default=False,
                        type=bool,
                        dest="unfroze_all")
    parser.add_argument('-m', '--model-name',
                        required=False,
                        default="MobileNetV2",
                        type=str,
                        dest="model_name")
    parser.add_argument('-d', '--dense-layer-size',
                        required=False,
                        nargs="*",
                        default=[],
                        type=int,
                        dest="dense_size")
    parser.add_argument('-is', '--input-size',
                        required=False,
                        default=96,
                        type=int,
                        dest="input_size")
    parser.add_argument('-viz', '--data-visualisation',
                        required=False,
                        default=False,
                        type=bool,
                        dest="data_visualisation")
    args = parser.parse_args()
    batch_size = args.batch

    class_name = args.train_class
    out_classes = classes[class_name]["signs_classes"]
    rotation_and_flips = classes[class_name]["rotation_and_flips"]
    h_symmetry_classes = classes[class_name]["h_symmetry"]
    try:
        merge_sign_classes = classes[class_name]["merge_sign_classes"]
    except KeyError:
        merge_sign_classes = None

    mapping = {c: i for i, c in enumerate(out_classes)}
    mapping_id_to_name = {i: c for c, i in mapping.items()}

    os.makedirs(class_name, exist_ok=True)

    x_train, y_train, x_test, y_test = get_data_for_master_class(class_name=class_name,
                                                                 mapping=mapping,
                                                                 mapping_id_to_name=mapping_id_to_name,
                                                                 rotation_and_flips=rotation_and_flips,
                                                                 data_dir=args.data_dir,
                                                                 merge_sign_classes=merge_sign_classes,
                                                                 h_symmetry_classes=h_symmetry_classes,
                                                                 image_size=(args.input_size, args.input_size),
                                                                 ignore_npz=args.ignore_npz,
                                                                 out_classes=out_classes)
    if args.data_visualisation:
        preprocess_input = lambda x: x
        model = None
    else:
        if args.random_init:
            weights = None
        else:
            weights = 'imagenet'
        if args.model_name == "MobileNetV2":
            preprocess_input = mobilenetv2.preprocess_input
            base_model = mobilenetv2.MobileNetV2(weights=weights,
                                                 include_top=False,
                                                 input_shape=(args.input_size, args.input_size, 3),
                                                 pooling='avg')
        elif args.model_name == "InceptionResNetV2":
            preprocess_input = inception_resnet_v2.preprocess_input
            base_model = inception_resnet_v2.InceptionResNetV2(weights=weights,
                                                               include_top=False,
                                                               input_shape=(args.input_size, args.input_size, 3),
                                                               pooling='avg')
        elif args.model_name == "NASNetLarge":
            preprocess_input = nasnet.preprocess_input
            base_model = nasnet.NASNetLarge(weights=weights,
                                            include_top=False,
                                            input_shape=(args.input_size, args.input_size, 3),
                                            pooling='avg')
        else:
            raise ValueError("unknown model name {}, should be one of {}".format(args.model_name,
                                                                                 ["MobileNetV2", "InceptionResNetV2",
                                                                                  "NASNetLarge"]))
        predictions = base_model.outputs[0]
        for s in args.dense_size:
            predictions = Dense(s, activation='relu')(predictions)
        predictions = Dense(len(out_classes), activation='softmax')(predictions)
        model = Model(inputs=base_model.input, outputs=predictions)

    # model.summary()
    # blocks = {}
    # for i, layer in enumerate(base_model.layers):
    #     s = layer.name.split('_')
    #     if s[0] == "block":
    #         b = int(s[1])
    #         if b not in blocks:
    #             blocks[b] = [i]
    #         else:
    #             blocks[b].append(i)
    # exit(0)

    callbacks = [ModelCheckpoint(filepath="{}/checkpoint.h5".format(class_name),
                                 monitor="val_loss",
                                 mode='min',
                                 verbose=0,
                                 save_best_only="True",
                                 save_weights_only=False,
                                 period=1),
                 EarlyStopping(monitor='val_acc',
                               mode='max',
                               min_delta=0.001,
                               patience=40,
                               verbose=1,
                               restore_best_weights=True)
                 ]

    x_test = np.stack([preprocess_input(i) for i in x_test])
    datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 brightness_range=(0.5, 1.4),
                                 shear_range=3.0,
                                 zoom_range=(0.7, 1.1),
                                 fill_mode='nearest',
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 preprocessing_function=preprocess_input)
    datagen.fit(x_train)

    if args.data_visualisation:
        for b in datagen.flow(x_train, y_train, batch_size=1):
            im, im_class = b[0][0], b[1][0]
            im_class = int(np.argmax(im_class))
            plt.imshow(im.astype(np.int))
            plt.title(out_classes[im_class])
            plt.show()
        return

    if not args.random_init:
        # if the network is not randomly initialized, we first fine tune the last layers
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer=rmsprop(lr=args.lr, decay=args.decay),
                      loss='categorical_crossentropy', metrics=["accuracy"])
        data = datagen.flow(x_train, y_train, batch_size=batch_size)
        data = custom_transform(data)

        history = model.fit_generator(data,
                                      steps_per_epoch=ceil(len(x_train) / batch_size),
                                      epochs=args.epoch,
                                      verbose=1,
                                      validation_data=(x_test, y_test),
                                      use_multiprocessing=True,
                                      callbacks=callbacks)
        plot_history(history, "{0}/{1}_{0}_dense_".format(class_name, args.model_name))
        model.save("{0}/{1}_{0}_dense.h5".format(class_name, args.model_name), overwrite=True)

        if not args.unfroze_all:
            # unfroze the 3 last blocks of mobile net
            for layer in model.layers[:113]:
                layer.trainable = False
            for layer in model.layers[113:]:
                layer.trainable = True
            model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, decay=args.decay),
                          loss='categorical_crossentropy', metrics=["accuracy"])
            data = datagen.flow(x_train, y_train, batch_size=batch_size)
            data = custom_transform(data)

            history = model.fit_generator(data,
                                          steps_per_epoch=ceil(len(x_train) / batch_size),
                                          epochs=args.epoch_fine_tune,
                                          verbose=1,
                                          validation_data=(x_test, y_test),
                                          use_multiprocessing=True,
                                          callbacks=callbacks)
            plot_history(history, "{0}/{1}_{0}_fine_tuning_1_".format(class_name, args.model_name))

            model.save("{0}/{1}_{0}_1.h5".format(class_name, args.model_name), overwrite=True)

            # unfroze the 6 last blocks of mobile net
            for layer in model.layers[:87]:
                layer.trainable = False
            for layer in model.layers[87:]:
                layer.trainable = True
            model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, decay=args.decay),
                          loss='categorical_crossentropy', metrics=["accuracy"])
            data = datagen.flow(x_train, y_train, batch_size=batch_size)
            data = custom_transform(data)

            history = model.fit_generator(data,
                                          steps_per_epoch=ceil(len(x_train) / batch_size),
                                          epochs=args.epoch_fine_tune,
                                          verbose=1,
                                          validation_data=(x_test, y_test),
                                          use_multiprocessing=True,
                                          callbacks=callbacks)
            plot_history(history, "{0}/{1}_{0}_fine_tuning_2_".format(class_name, args.model_name))

            model.save("{0}/{1}_{0}_2.h5".format(class_name, args.model_name), overwrite=True)

    # unfroze all model
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, decay=args.decay),
                  loss='categorical_crossentropy', metrics=["accuracy"])
    data = datagen.flow(x_train, y_train, batch_size=batch_size)
    data = custom_transform(data)

    history = model.fit_generator(data,
                                  steps_per_epoch=ceil(len(x_train) / batch_size),
                                  epochs=args.epoch_fine_tune,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  use_multiprocessing=True,
                                  callbacks=callbacks)
    plot_history(history, "{0}/{1}_{0}_fine_tuning_f_".format(class_name, args.model_name))

    model.save("{0}/{1}_{0}_final.h5".format(class_name, args.model_name), overwrite=True)


if __name__ == '__main__':
    main()


