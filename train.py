import imghdr
import itertools
import math
import os

import keras
import keras.backend as K
import numpy as np
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from keras.utils import generic_utils

from model import (build_discriminator_proxy, build_global_discriminator,
                   build_local_discriminator, build_model, bulid_generator)


def generate_padded_images(imgs, padding_width):
    padded_imgs = np.copy(imgs)
    pix_avg = np.mean(imgs, axis=(1, 2, 3))
    padded_imgs[:, :, :padding_width, :] = padded_imgs[:,
                                                       :, -padding_width:, :] = pix_avg.reshape(-1, 1, 1, 1)
    return padded_imgs


def crop_and_resize_image(img, img_size):
    source_size = img.size
    if source_size == img_size:
        return img
    max_size = min(source_size)
    img = img.crop([(img.width - max_size) // 2, (img.height - max_size) // 2,
                    (img.width - max_size) // 2 + max_size, (img.height - max_size) // 2 + max_size])
    img = img.resize(img_size)
    return img


class DataGenerator(object):
    def __init__(self, root_dir, img_size=(256, 256), batch_size=10, padding_width=64, validation_rate=0.1):
        if not isinstance(validation_rate, float) and not 0 <= validation_rate <= 1:
            raise ValueError()

        self.batch_size = batch_size
        self.img_size = img_size
        self.padding_width = padding_width

        self.reset()
        self.img_file_list = []

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if imghdr.what(full_path) is None:
                    continue
                self.img_file_list.append(full_path)

        validation_size = math.floor(validation_rate * len(self.img_file_list))
        self.validation_file_list = self.img_file_list[:validation_size]
        self.img_file_list = self.img_file_list[validation_size:]

    def __len__(self):
        return len(self.img_file_list)

    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    def get_file_generator(self, file_list, shuffle=True):
        while True:
            if shuffle:
                 np.random.shuffle(file_list)
            for f in file_list:
                img = crop_and_resize_image(load_img(f), self.img_size)
                self.images.append(img_to_array(img))

                if len(self.images) == self.batch_size:
                    imgs = (np.asarray(self.images,
                                       dtype=np.float32) / 255) * 2 - 1
                    self.reset()
                    yield generate_padded_images(imgs, self.padding_width), imgs

    def flow(self):
        return self.get_file_generator(self.img_file_list)

    def validation_flow(self, validation_steps=1):
        # def reshape_list(x, y):
        #     x[0].append(y[0])
        #     x[1].append(y[1])
        #     return x
        # data = functools.reduce(reshape_list, validation_generator, ([], []))
        # data = np.concatenate(data[0]), np.concatenate(data[1])
        return self.get_file_generator(self.validation_file_list, shuffle=False)


path = 'place365'


def train(path, batch_size=10, epochs=50, steps_per_epoch=30):
    data_generator = DataGenerator(path, batch_size)
    data_size = len(data_generator)
    data = data_generator.flow()

    G, D, C = build_model()

    t1_epochs = epochs * 18 // 100
    t2_epochs = epochs * 2 // 100
    t3_epochs = epochs * 80 // 100

    print('Phase 1')
    G.fit_generator(data, epochs=t1_epochs, steps_per_epoch=steps_per_epoch)
    a = G.predict_generator(data, steps=1)
    array_to_img(a[0]).show()

    print('Phase 2')
    for cur_epoch in range(t2_epochs):
        print('Epoch {}/{}'.format(cur_epoch, t2_epochs))
        progbar = generic_utils.Progbar(steps_per_epoch)
        for d in itertools.islice(data, None, steps_per_epoch):
            padded_iamges, real_images = d
            generated_images = G.predict(padded_iamges)
            combined_images = np.concatenate([generated_images, real_images])
            labels = np.concatenate([np.ones((batch_size, 1)),
                                     np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)
            d_loss = D.train_on_batch(combined_images, labels)
            progbar.add(1, values=[("D loss", d_loss)])

    print('Phase 3')
    for cur_epoch in range(t3_epochs):
        print('Epoch {}/{}'.format(cur_epoch, t3_epochs))
        progbar = generic_utils.Progbar(steps_per_epoch)
        for d in itertools.islice(data, None, steps_per_epoch):
            padded_iamges, real_images = d
            c_loss = C.train_on_batch(
                padded_iamges, [real_images, np.zeros(batch_size)])
            generated_images = G.predict(padded_iamges)
            combined_images = np.concatenate([generated_images, real_images])
            labels = np.concatenate([np.ones((batch_size, 1)),
                                     np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)
            d_loss = D.train_on_batch(combined_images, labels)
            progbar.add(1, values=[("D loss", d_loss),
                                   ('C loss', c_loss[0]), ('G loss', c_loss[1])])

    a = G.predict_generator(data, steps=1)
    array_to_img(a[0]).show()


def overfit_a_img(path):
    data_generator = DataGenerator(path)
    data_size = len(data_generator)
    data = next(data_generator.flow(1))

    array_to_img(data[0][0]).show()
    array_to_img(data[1][0]).show()

    G, D, C = build_model()

    G.fit(data[0], data[1], epochs=10, steps_per_epoch=100)
    a = G.predict(data[0])
    array_to_img(a[0]).show()

    G.fit(data[0], data[1], epochs=10, steps_per_epoch=100)
    a = G.predict(data[0])
    array_to_img(a[0]).show()


train(path)
# overfit_a_img(path)
# # G, D, C = build_model()
# # from keras.utils import plot_model
# # plot_model(G, to_file='modelg.png', show_shapes=True)
# # plot_model(D, to_file='modeld.png', show_shapes=True)
# # plot_model(C, to_file='modelc.png', show_shapes=True)
