import imghdr
import itertools
import math
import os
import json
from PIL import Image

import matplotlib.pyplot as plt
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


path = r'data\beach_image'


def plot_history(loss_history, model_name):
    plt.plot(loss_history['loss'])
    plt.plot(loss_history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('log/{}'.format(model_name), transparent=True)
    plt.close()


def combine_image_and_label(G, padded_iamges, real_images, batch_size):
    generated_images = G.predict(padded_iamges)
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.zeros((batch_size, 1), dtype='float32'),
                             np.ones((batch_size, 1), dtype='float32')])
    #labels += 0.05 * np.random.random(labels.shape)

    return combined_images, labels


def plot_generated_image(G, data_generator):
    padded_inputs, inputs = next(data_generator.validation_flow())
    generated_images = G.predict(padded_inputs)
    plt.figure(figsize=(3, 6))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow((generated_images[i] + 1) / 2)
        plt.subplot(5, 2, 2 * i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow((inputs[i] + 1) / 2)
    plt.savefig('log/{}'.format('test'), transparent=True)
    plt.close()


def get_counter():
    counter = 0

    def inner_func():
        nonlocal counter
        counter += 1
        if counter == 20:
            counter = 0
            return True
        return False
    return inner_func


def train(path, batch_size=10, epochs=50, steps_per_epoch=None):
    # input_shape = (256, 256, 3)
    # padding_width = 64
    input_shape = (128, 128, 3)
    padding_width = 32
    is_log = get_counter()

    data_generator = DataGenerator(
        path, input_shape[:2], batch_size, padding_width)
    data_size = len(data_generator)
    print('data size: {}'.format(data_size))

    if steps_per_epoch is None:
        steps_per_epoch = data_size // batch_size

    G, D, C = build_model(input_shape, padding_width)

    t1_epochs = epochs * 16 // 100
    t2_epochs = epochs * 2 // 100
    t3_epochs = epochs * 70 // 100

    g_history = {'loss': [], 'val_loss': []}
    d_history = {'loss': [], 'val_loss': []}
    c_history = {'loss': [], 'val_loss': []}

    class LogCallback(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if is_log():
                g_history['loss'].append(logs['loss'])
                padded_iamges, real_images = next(
                    data_generator.validation_flow())
                val_loss = G.evaluate(padded_iamges, real_images, batch_size)
                g_history['val_loss'].append(val_loss)
                plot_history(g_history, 'G')
                plot_generated_image(G, data_generator)

    print('Phase 1')
    if os.path.exists('checkpoint/g.h5'):
        G.load_weights('checkpoint/g.h5')
        g_history = json.load(
            open('checkpoint/g_history.json', 'r', encoding='utf-8'))
        print('get trained G weight')
    else:
        g_loss = G.fit_generator(data_generator.flow(), epochs=t1_epochs, steps_per_epoch=steps_per_epoch,
                                 validation_data=data_generator.validation_flow(), validation_steps=2, callbacks=[LogCallback()])
        G.save_weights('checkpoint/g.h5')
        g_history = g_loss.history
        json.dump(g_history, open(
            'checkpoint/g_history.json', 'w', encoding='utf-8'))

    json.dump(c_history, open(
        'checkpoint/c_history.json', 'w', encoding='utf-8'))
    json.dump(d_history, open(
        'checkpoint/d_history.json', 'w', encoding='utf-8'))
    json.dump(g_history, open(
        'checkpoint/g_end_history.json', 'w', encoding='utf-8'))

    C.save_weights('checkpoint/c.h5')
    D.save_weights('checkpoint/d.h5')
    G.save_weights('checkpoint/g_end.h5')

    print('Phase 2')
    counter = 0
    D.summary()
    for cur_epoch in range(t2_epochs):
        print('Epoch {}/{}'.format(cur_epoch, t2_epochs))
        progbar = generic_utils.Progbar(steps_per_epoch)
        for d in itertools.islice(data_generator.flow(), None, steps_per_epoch):
            padded_iamges, real_images = d
            fake_images = G.predict(padded_iamges)
            d_loss_real = D.train_on_batch(
                real_images, np.ones(batch_size, dtype='float32'))
            d_loss_fake = D.train_on_batch(
                fake_images, np.zeros(batch_size, dtype='float32'))
            d_loss = (d_loss_real + d_loss_fake) / 2
            progbar.add(1, values=[("D loss", d_loss)])
            if is_log():
                d_history['loss'].append(float(d_loss))
                padded_iamges, real_images = next(
                    data_generator.validation_flow())
                combined_images, labels = combine_image_and_label(
                    G, padded_iamges, real_images, batch_size)
                d_val_loss = D.evaluate(combined_images, labels)
                d_history['val_loss'].append(float(d_val_loss))
                plot_history(d_history, 'D')

    print('Phase 3')
    for cur_epoch in range(t3_epochs):
        print('Epoch {}/{}'.format(cur_epoch, t3_epochs))
        progbar = generic_utils.Progbar(steps_per_epoch)
        for d in itertools.islice(data_generator.flow(), None, steps_per_epoch):
            padded_iamges, real_images = d

            fake_images = G.predict(padded_iamges)
            d_loss_real = D.train_on_batch(
                real_images, np.ones(batch_size, dtype='float32'))
            d_loss_fake = D.train_on_batch(
                fake_images, np.zeros(batch_size, dtype='float32'))
            d_loss = (d_loss_real + d_loss_fake) / 2

            c_loss = C.train_on_batch(
                padded_iamges, [real_images, np.ones((batch_size, 1), dtype='float32')])

            progbar.add(1, values=[("D loss", d_loss),
                                   ('C loss', c_loss[0]), ('G loss', c_loss[1])])
            if is_log():
                d_history['loss'].append(float(d_loss))
                c_history['loss'].append(float(c_loss[0]))
                g_history['loss'].append(float(c_loss[1]))
                padded_iamges, real_images = next(
                    data_generator.validation_flow())
                c_loss = C.evaluate(
                    padded_iamges, [real_images, np.ones((batch_size, 1), dtype='float32')])
                combined_images, labels = combine_image_and_label(
                    G, padded_iamges, real_images, batch_size)
                d_loss = D.evaluate(combined_images, labels)
                d_history['val_loss'].append(float(d_loss))
                c_history['val_loss'].append(float(c_loss[0]))
                g_history['val_loss'].append(float(c_loss[1]))
                plot_history(d_history, 'D')
                plot_history(c_history, 'C')
                plot_history(g_history, 'G')
                plot_generated_image(G, data_generator)

    G.save_weights('checkpoint/g_end.h5')
    json.dump(g_history, open(
        'checkpoint/g_end_history.json', 'w', encoding='utf-8'))
    D.save_weights('checkpoint/d.h5')
    json.dump(d_history, open(
        'checkpoint/d_history.json', 'w', encoding='utf-8'))


def overfit_a_img(path):
    img = Image.open('test/a.jpg')

    img = (
        np.array([img_to_array(crop_and_resize_image(img, (128, 128)))]) / 255) * 2 - 1
    padded_img = generate_padded_images(img, 32)

    array_to_img(img[0]).save('test/aa.jpg')
    array_to_img(padded_img[0]).save('test/pad.jpg')

    G, D, C = build_model((128, 128, 3), 32)

    G.fit(padded_img, img, epochs=10, steps_per_epoch=100)
    a = G.predict(padded_img)
    array_to_img((a[0] + 1) / 2).save('test/888.jpg')


def generate_all_val():
    data_generator = DataGenerator(path, (128, 128), 10, 32)
    data_size = len(data_generator)

    print(len(data_generator.validation_file_list),
          len(data_generator.img_file_list))

    # img1 = Image.open('test/a.jpg')
    # img2 = Image.open('test/b.jpg')

    # imgs = (np.array([
    #     img_to_array(crop_and_resize_image(img1, (128, 128))),
    #     img_to_array(crop_and_resize_image(img2, (128, 128)))
    # ]) / 255) * 2 - 1
    # imgs = generate_padded_images(imgs, 32)

    # array_to_img(imgs[0]).show()

    G, D, C = build_model((128, 128, 3), 32)
    G.load_weights('checkpoint/g.h5')

    # a = G.predict(imgs)
    # array_to_img((a[0] + 1) / 2).save('test/a_gen.jpg')
    # array_to_img((a[1] + 1) / 2).save('test/b_gen.jpg')

    for ii, val in enumerate(data_generator.validation_flow()):
        a = G.predict(val[0])

        for i, img in enumerate(zip(val[1], a)):
            array_to_img((img[0] + 1) /
                         2).save('output\{}-{}.jpg'.format(ii, i))
            array_to_img(
                img[1] + 1 / 2).save('output\{}-{}_gen.jpg'.format(ii, i))


# def generate_padded_images(imgs, padding_width):
#     padded_imgs = np.copy(imgs)
#     pix_avg = np.mean(imgs, axis=(1, 2, 3))
#     padded_imgs[:, :, :padding_width, :] = padded_imgs[:,
#                                                        :, -padding_width:, :] = pix_avg.reshape(-1, 1, 1, 1)
#     return padded_imgs


def pressure_test(t=3, padding_width=32):
    img1 = Image.open('test/a.jpg')
    img2 = Image.open('test/b.jpg')

    imgs = (np.array([
        img_to_array(crop_and_resize_image(img1, (128, 128))),
        img_to_array(crop_and_resize_image(img2, (128, 128)))
    ]) / 255) * 2 - 1

    G, D, C = build_model((128, 128, 3), padding_width)
    G.load_weights('checkpoint/g_end.h5')

    for _ in range(t):
        mid_width = 64
        new_shape = list(imgs.shape)
        new_shape[2] += padding_width * 2
        new_imgs = np.zeros(new_shape)
        new_imgs[:, :, padding_width:-padding_width, :] = imgs

        left_imgs = np.zeros((new_shape[0], 128, 128, 3))
        right_imgs = np.zeros((new_shape[0], 128, 128, 3))

        pix_avg = np.mean(imgs[:, :, :mid_width, :], axis=(1, 2, 3))
        left_imgs[:, :, padding_width:-
                  padding_width, :] = imgs[:, :, :mid_width, :]
        left_imgs[:, :, :padding_width, :] = left_imgs[:, :, -padding_width:, :] = pix_avg.reshape(-1, 1, 1, 1)
        
        a = G.predict(left_imgs)
        new_imgs[:, :, :padding_width, :] =  a[:, :, :padding_width, :]
        

        right_imgs[:, :, padding_width:-
                   padding_width, :] = imgs[:, :, -mid_width:, :]
        pix_avg = np.mean(imgs[:, :, -mid_width:, :], axis=(1, 2, 3))
        right_imgs[:, :, :padding_width, :] = right_imgs[:, :, -padding_width:, :] = pix_avg.reshape(-1, 1, 1, 1)

        a = G.predict(right_imgs)
        new_imgs[:, :, -padding_width:, :] =  a[:, :, -padding_width:, :]

        imgs = new_imgs

    array_to_img(imgs[0]).save('test/a_full.jpg')
    array_to_img(imgs[1]).save('test/b_full.jpg')

        



#train(path)
# print('ok! shutdown in 60s')
# import time
# time.sleep(60)
# os.system()
generate_all_val()
# pressure_test()