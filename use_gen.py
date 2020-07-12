
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def create_plot(examples, n):
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(np.squeeze(examples[i, :, :]), cmap="gray")
    filename = "fake_images.png"
    pyplot.savefig(filename, cmap="gray")
    pyplot.close()

if __name__ == "__main__":
    model = load_model("saved_model/g_model.h5")
    latent_points = generate_latent_points(100, 25)
    x = model.predict(latent_points)
    x = (x + 1) / 2.0
    create_plot(x, 5)
