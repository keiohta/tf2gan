import argparse
import numpy as np
import tensorflow as tf

from tf2gan.gan import GAN
from tf2gan.networks.simple_net import Generator, Discriminator

if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


def get_mnist_data(flatten=False, max_size=None):
    print("[info] loading mnist image...")
    mnist = tf.keras.datasets.mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    xs = np.concatenate((x_train, x_test), axis=0)  # 70000x28x28

    if flatten:
        xs = xs.reshape(xs.shape[0], -1)
    else:
        xs = np.pad(xs, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.)  # 70000x32x32
        xs = np.expand_dims(xs, axis=xs.ndim)  # 70000x32x32x1
    return xs if max_size is None else xs[:max_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--test-freq', type=int, default=10)
    parser.add_argument('--max-size', type=int, default=10000)
    args = parser.parse_args()

    real_imgs = get_mnist_data(flatten=False, max_size=args.max_size)
    assert real_imgs.ndim == 4

    G = Generator(img_size=real_imgs.shape[1:])
    D = Discriminator(img_size=real_imgs.shape[1:])

    gan = GAN(generator=G, discriminator=D)
    gan.train(real_imgs, n_epochs=args.n_epochs, test_freq=args.test_freq)


if __name__ == "__main__":
    main()
