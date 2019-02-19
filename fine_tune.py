import matplotlib
# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from math import ceil
import numpy as np

from train import plot_history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("weightsIn", type=str)
    parser.add_argument("weightsOut", type=str)
    parser.add_argument("epoch", type=int)
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
    parser.add_argument('-u', '--unfroze-block-count',
                        required=False,
                        default=0,
                        type=int,
                        dest="unfroze_block_count")
    args = parser.parse_args()

    batch_size = 1024
    savez = np.load("data.npz")
    x_train = savez["x_train"]
    y_train = savez["y_train"]
    x_test = savez["x_test"]
    y_test = savez["y_test"]
    out_classes = savez["out_classes"]
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(96, 96, 3),
                             pooling='avg')
    predictions = Dense(len(out_classes), activation='softmax')(base_model.outputs[0])
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(args.weightsIn)

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

    # count blocks and froze layers
    blocks = {}
    for i, layer in enumerate(base_model.layers):
        s = layer.name.split('_')
        if s[0] == "block":
            b = int(s[1])
            if b not in blocks:
                blocks[b] = [i]
            else:
                blocks[b].append(i)
    for b, layers_id in blocks.items():
        if b > len(blocks) - args.unfroze_block_count:
            for layer_id in layers_id:
                base_model.layers[layer_id].trainable = True
        else:
            for layer_id in layers_id:
                base_model.layers[layer_id].trainable = False

    # compile and train
    model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, decay=args.decay),
                  loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch),
                                  steps_per_epoch=ceil(len(x_train) / args.batch),
                                  epochs=args.epoch,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  use_multiprocessing=True)
    plot_history(history, args.weightsOut)

    model.save(args.weightsOut, overwrite=True)





