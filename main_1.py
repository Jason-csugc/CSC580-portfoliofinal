"""TensorFlow/Keras GAN training script for a single CIFAR-10 class."""

import os
import logging
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Disable oneDNN optimizations for consistent performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Ignore NumPy 2.4 deprecation warning triggered by CIFAR-10 pickle loading.
warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as Python or NumPy boolean.*",
    category=Warning,
)

import tensorflow as tf
from keras import layers, Sequential, Model
from keras.datasets import cifar10
from keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SHAPE = (32, 32, 3)
LATENT_DIM = 100
CLASS_ID = 8          # CIFAR-10 class 8 = ship
EPOCHS = 15000
BATCH_SIZE = 32
DISPLAY_INTERVAL = 2500
OUTPUT_DIR = "gan_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_cifar10_class(class_id=8):
    """Load CIFAR-10 and return normalized samples for a single class."""
    (x_train, y_train), (_, _) = cifar10.load_data()

    x_train = x_train[y_train.flatten() == class_id]

    # Normalize from [0, 255] to [-1, 1]
    x_train = (x_train.astype("float32") / 127.5) - 1.0

    return x_train


def build_generator():
    """Construct and return the GAN generator network."""
    model = Sequential(name="Generator")

    model.add(layers.Input(shape=(LATENT_DIM,)))
    model.add(layers.Dense(128 * 8 * 8, activation="relu"))
    model.add(layers.Reshape((8, 8, 128)))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.78))
    model.add(layers.Activation("relu"))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.78))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(layers.Activation("tanh"))

    return model


def build_discriminator():
    """Construct and return the GAN discriminator network."""
    model = Sequential(name="Discriminator")

    model.add(layers.Input(shape=IMAGE_SHAPE))

    model.add(layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(layers.BatchNormalization(momentum=0.82))
    model.add(layers.LeakyReLU(negative_slope=0.25))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.82))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(negative_slope=0.25))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def save_generated_images(generator, epoch_label, filename):
    """Generate a 4x4 image grid and display it for the current epoch."""
    rows, cols = 4, 4

    noise = np.random.normal(0, 1, (rows * cols, LATENT_DIM))
    generated_images = generator.predict(noise, verbose=0)

    # Convert from [-1, 1] back to [0, 1]
    generated_images = 0.5 * generated_images + 0.5
    generated_images = np.clip(generated_images, 0, 1)

    _, axs = plt.subplots(rows, cols, figsize=(6, 6))
    count = 0

    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis("off")
            count += 1

    plt.suptitle(f"Generated CIFAR-10 Images - {epoch_label}")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    # plt.savefig(save_path, dpi=150, format="jpg")
    plt.show()
    plt.close()

    print(f"Saved image grid: {save_path}")


def train_gan():
    """Train the GAN and visualize generated images and training losses."""
    print(matplotlib.get_backend())
    x_train = load_cifar10_class(CLASS_ID)

    generator = build_generator()
    discriminator = build_discriminator()

    # Compile discriminator while trainable
    discriminator.trainable = True
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"]
    )

    # Freeze discriminator only for combined GAN
    discriminator.trainable = False

    z = layers.Input(shape=(LATENT_DIM,))
    generated_image = generator(z)
    validity = discriminator(generated_image)

    combined_network = Model(z, validity, name="Combined_GAN")
    combined_network.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
    )

    d_losses = []
    g_losses = []

    for epoch in range(1, EPOCHS + 1):

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        discriminator.trainable = True

        index = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
        real_images = x_train[index]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        generated_images = generator.predict(noise, verbose=0)

        valid = np.ones((BATCH_SIZE, 1))
        valid += 0.05 * np.random.random(valid.shape)

        fake = np.zeros((BATCH_SIZE, 1))
        fake += 0.05 * np.random.random(fake.shape)

        disc_loss_real = discriminator.train_on_batch(real_images, valid)
        disc_loss_fake = discriminator.train_on_batch(generated_images, fake)
        disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

        # -----------------------------
        # Train Generator
        # -----------------------------
        discriminator.trainable = False

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        misleading_targets = np.ones((BATCH_SIZE, 1))

        gen_loss = combined_network.train_on_batch(noise, misleading_targets)

        d_losses.append(disc_loss[0])
        g_losses.append(gen_loss)

        if epoch == 1:
            save_generated_images(
                generator,
                "Epoch 1",
                "cifar10_epoch_1.jpg"
            )

        if epoch % DISPLAY_INTERVAL == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| D Loss: {disc_loss[0]:.4f} "
                f"| D Accuracy: {disc_loss[1] * 100:.2f}% "
                f"| G Loss: {gen_loss:.4f}"
            )

        if epoch == EPOCHS:
            save_generated_images(
                generator,
                f"Final Epoch {EPOCHS}",
                "cifar10_final_epoch.jpg"
            )

    # Save loss curve as JPG
    plt.figure(figsize=(8, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("CIFAR-10 GAN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(OUTPUT_DIR, "cifar10_gan_loss_curve.jpg")
    # plt.savefig(loss_path, dpi=150, format="jpg")
    plt.show()
    plt.close()

    print(f"Saved loss curve: {loss_path}")


if __name__ == "__main__":
    train_gan()