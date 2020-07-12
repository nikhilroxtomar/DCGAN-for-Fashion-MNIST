
import os
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

def build_discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    ## 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    ## 2
    model.add(Conv2D(64, (3, 3), padding="same", strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    ## 3
    model.add(Conv2D(128, (3, 3), padding="same", strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    ## 4
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    model.load_weights("saved_model/d_model.h5")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['acc'])
    return model

def build_generator(latent_dim):
    model = Sequential()
    nodes = 128 * 7 * 7
    ## 1
    model.add(Dense(nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    ## 2 (14x14)
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    ## 3 (28x28)
    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    ## 4 (28x28x1)
    model.add(Conv2D(1, (3, 3), padding="same", activation="tanh"))

    model.load_weights("saved_model/g_model.h5")
    return model

def build_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model

def load_real_samples():
    (trainx, _), (testx, _) = fashion_mnist.load_data()
    data = np.concatenate([trainx, testx], axis=0)
    data = np.expand_dims(data, axis=-1)
    x = data.astype(np.float32)
    x = (x - 127.5) / 127.5
    return x

def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = np.ones((n_samples, 1))
    return x, y

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    x = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return x, y

def save_plot(examples, epoch, n=5):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(np.squeeze(examples[i]), cmap="gray")
    filename = f"samples/generated_plot_epoch-{epoch+1}.png"
    pyplot.savefig(filename, cmap="gray")
    pyplot.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    x_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(">Acc Real: {:1.4f} - Acc Fake: {:1.4f}".format(acc_real, acc_fake))

    save_plot(x_fake, epoch)
    g_model.save(f"saved_model/g_model.h5")
    d_model.save(f"saved_model/d_model.h5")

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    batch_per_epoch = dataset.shape[0]//n_batch
    half_batch = n_batch // 2

    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(x_real, y_real)

            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(x_fake, y_fake)

            x_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(x_gan, y_gan)

            print(">{:1.0f} - {:1.0f}/{:1.0f} - D1_loss: {:1.4f} - D2_loss: {:1.4f} - G_loss: {:1.4f}".format(
                i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss
            ))

        if (i + 1) % 10:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

if __name__ == "__main__":
    ## Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    ##
    latent_dim = 100
    n_batch = 150
    n_epochs = 100

    d_model = build_discriminator()
    g_model = build_generator(latent_dim)
    gan_model = build_gan(g_model, d_model)
    dataset = load_real_samples()
    train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=n_epochs, n_batch=n_batch)
