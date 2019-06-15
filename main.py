import numpy as np
from keras.layers import (Dense, Dropout, Input, Conv2D, Reshape, Flatten, LeakyReLU, multiply,
                          Embedding, BatchNormalization, UpSampling2D, Activation, ReLU)
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32)-127.5)/127.5
x_train = np.expand_dims(x_train, axis=3)
y_train = y_train.reshape(-1, 1)


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def create_generator(num_batches=128):
    model = Sequential()
    random_input = Input(shape=(100,))
    num_input = Input(shape=(1,), dtype='uint8')
    num_input_l = Flatten()(Embedding(10, 100)(num_input))
    model.add(Dense(128 * 7 * 7, input_dim=100, activation='relu'))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, (3, 3), padding="same"))
    model.add(Activation('tanh'))
    inp = multiply([random_input, num_input_l])
    img = model(inp)
    generator = Model([random_input, num_input], img)
    # generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator


def create_discriminator(num_batches=128):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=2, padding="same", input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    img = Input(shape=(28, 28, 1))
    features = model(img)
    validity = Dense(1, activation='sigmoid')(features)
    label = Dense(1, activation='softmax')(features)
    discriminator = Model(img, [validity, label])
    discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                          optimizer=adam_optimizer())
    return discriminator


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input_R = Input(shape=(100,))
    gan_input_N = Input(shape=(1,))
    x = generator([gan_input_R, gan_input_N])
    valid, target = discriminator(x)
    gan = Model(inputs=[gan_input_R, gan_input_N], output=[valid, target])
    gan.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                optimizer=adam_optimizer())
    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    rint = np.random.randint(0, 10, [examples, 1])
    generated_images = generator.predict([noise, rint])
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.title(rint[i][0])
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_%d.png' % epoch)


def training(epochs=1, batch_size=128):
    global x_train, y_train, x_test, y_test

    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)
    valid = np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))

    for e in range(1, epochs+1):
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_size)):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            labels = y_train[idx]
            noise = np.random.normal(0, 1, [batch_size, 100])
            fake_labels = np.random.randint(0, 10, [batch_size, 1])
            generated_imgs = generator.predict([noise, fake_labels])
            discriminator.trainable = True
            discriminator.train_on_batch(imgs, [valid, labels])
            discriminator.train_on_batch(generated_imgs, [fake, fake_labels])
            discriminator.trainable = False
            gan.train_on_batch([noise, fake_labels], [valid, fake_labels])

        if e == 1 or e % 10 == 0:
            plot_generated_images(e, generator)


training(400, 32)
