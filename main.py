"""
TensorFlow 2 / Keras GAN for CIFAR-10

Generates CIFAR-10-style images for one selected class.
Saves:
    generated_epoch_1.png
    generated_final_epoch.png
"""

import os
import warnings
from time import time
import numpy as np
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

# pylint: disable=wrong-import-position
import tensorflow as tf
import keras as ks

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Parameters
# -----------------------------
IMAGE_SHAPE = (32, 32, 3)
LATENT_DIM = 100
CLASS_ID = 8         # CIFAR-10 class 8 = ship
EPOCHS = 15000
BATCH_SIZE = 64
DISPLAY_INTERVAL = 250
OUTPUT_DIR = "gan_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_cifar10_class(class_id=8):
    """Load CIFAR-10 and return normalized samples for a single class."""
    (x_train, y_train), (_, _) = ks.datasets.cifar10.load_data()

    # Select one class only
    x_train = x_train[y_train.flatten() == class_id]

    # Normalize images from [0, 255] to [-1, 1]
    x_train = (x_train.astype("float32") / 127.5) - 1.0

    return x_train


# def build_generator():
#     """Construct and return the GAN generator network."""
#     model = ks.Sequential(name="Generator")

#     model.add(ks.layers.Input(shape=(LATENT_DIM,)))
#     model.add(ks.layers.Dense(128 * 8 * 8, activation="relu"))
#     model.add(ks.layers.Reshape((8, 8, 128)))

#     model.add(ks.layers.UpSampling2D())
#     model.add(ks.layers.Conv2D(128, kernel_size=3, padding="same"))
#     model.add(ks.layers.BatchNormalization(momentum=0.78))
#     model.add(ks.layers.Activation("relu"))

#     model.add(ks.layers.UpSampling2D())
#     model.add(ks.layers.Conv2D(64, kernel_size=3, padding="same"))
#     model.add(ks.layers.BatchNormalization(momentum=0.78))
#     model.add(ks.layers.Activation("relu"))

#     model.add(ks.layers.Conv2D(3, kernel_size=3, padding="same"))
#     model.add(ks.layers.Activation("tanh"))

#     return model
def build_generator():
    model = ks.Sequential(name="Generator")

    model.add(ks.layers.Input(shape=(LATENT_DIM,)))

    # Project latent vector into low-resolution feature map
    model.add(ks.layers.Dense(4 * 4 * 512, use_bias=False))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    model.add(ks.layers.Reshape((4, 4, 512)))

    # 4x4 -> 8x8
    model.add(ks.layers.Conv2DTranspose(
        256,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    ))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    # 8x8 -> 16x16
    model.add(ks.layers.Conv2DTranspose(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    ))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    # 16x16 -> 32x32
    model.add(ks.layers.Conv2DTranspose(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    ))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    # Final RGB image output: 32x32x3
    model.add(ks.layers.Conv2DTranspose(
        3,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="tanh"
    ))

    return model

def build_discriminator():
    """Construct and return the GAN discriminator network."""
    model = ks.Sequential(name="Discriminator")

    model.add(ks.layers.Input(shape=IMAGE_SHAPE))

    model.add(ks.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ks.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(ks.layers.BatchNormalization(momentum=0.82))
    model.add(ks.layers.LeakyReLU(negative_slope=0.25))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(ks.layers.BatchNormalization(momentum=0.82))
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(ks.layers.BatchNormalization(momentum=0.8))
    model.add(ks.layers.LeakyReLU(negative_slope=0.25))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(1, activation="sigmoid"))

    return model


def save_generated_images(generator, epoch_label, filename):
    """Generate a 4x4 image grid and save it to the output directory."""
    rows, cols = 4, 4
    noise = np.random.normal(0, 1, (rows * cols, LATENT_DIM))
    generated_images = generator.predict(noise, verbose=0)

    # Rescale from [-1, 1] to [0, 1]
    generated_images = 0.5 * generated_images + 0.5
    generated_images = np.clip(generated_images, 0, 1)

    _, axs = plt.subplots(rows, cols, figsize=(6, 6))
    count = 0

    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis("off")
            count += 1

    plt.suptitle(f"Generated Images - {epoch_label}")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.close('all')

    print(f"Saved image grid: {save_path}")


def train():
    """Train the GAN on one CIFAR-10 class and save progress artifacts."""
    x_train = load_cifar10_class(CLASS_ID)

    start_time = time()

    generator = build_generator()
    print("--------------Generator Summary--------------")
    generator.summary()
    discriminator = build_discriminator()
    print("--------------Discriminator Summary--------------")
    discriminator.summary()

    discriminator.trainable = True
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=ks.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
        metrics=["accuracy"]
    )

    # Combined GAN model
    discriminator.trainable = False

    z = ks.layers.Input(shape=(LATENT_DIM,))
    generated_image = generator(z)
    validity = discriminator(generated_image)

    combined_network = ks.Model(z, validity, name="Combined_GAN")
    combined_network.compile(
        loss="binary_crossentropy",
        optimizer=ks.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )

    discriminator.trainable = True

    d_losses = []
    g_losses = []

    for epoch in range(1, EPOCHS + 1):

        # -------------------------
        # Train Discriminator
        # -------------------------
        idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
        real_images = x_train[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_images = generator.predict(noise, verbose=0)

        # Label smoothing
        valid = np.ones((BATCH_SIZE, 1)) * 0.9
        fake = np.zeros((BATCH_SIZE, 1))

        discriminator.trainable = True

        # d_loss_real = discriminator.train_on_batch(real_images, valid)
        # d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        for _ in range(2):  # or even 3
            disc_loss_real = discriminator.train_on_batch(real_images, valid)
            disc_loss_fake = discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

        # -------------------------
        # Train Generator
        # -------------------------
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

        # Generator wants discriminator to classify fake images as real
        misleading_targets = np.ones((BATCH_SIZE, 1))

        discriminator.trainable = False

        g_loss = combined_network.train_on_batch(noise, misleading_targets)

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        # Save first epoch images
        if epoch == 1:
            save_generated_images(
                generator,
                "Epoch 1",
                "generated_epoch_1.png"
            )

        # Display progress
        if epoch % DISPLAY_INTERVAL == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| D Loss: {d_loss[0]:.4f} "
                f"| D Accuracy: {d_loss[1] * 100:.2f}% "
                f"| G Loss: {g_loss:.4f}"
            )
            save_generated_images(
                generator,
                f"Epoch {epoch}",
                f"generated_epoch_{epoch}.png"
            )

        # Save final epoch images
        if epoch == EPOCHS:
            save_generated_images(
                generator,
                f"Final Epoch {EPOCHS}",
                "generated_final_epoch.png"
            )

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("GAN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(OUTPUT_DIR, "gan_loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.show(block=False)
    plt.close('all')

    print(f"Saved loss curve: {loss_path}")

    hours, rem = divmod(time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


if __name__ == "__main__":
    train()
