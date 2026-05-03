"""Train a CIFAR-10 class-conditional GAN and export training artifacts.

This module builds and trains a GAN that learns samples from a single
CIFAR-10 class and saves generated image grids, a loss plot, and a training
progress GIF.
"""

import os
import warnings
from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

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
USE_SINGLE_CLASS = True  # Set to True to train on only one class, False to use all classes

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_cifar10_data():
    """Load CIFAR-10 and optionally filter by class."""
    (x_train, y_train), (_, _) = ks.datasets.cifar10.load_data()

    if USE_SINGLE_CLASS:
        x_train = x_train[y_train.flatten() == CLASS_ID]

    # Normalize to [-1, 1]
    x_train = x_train.astype("float32")
    x_train = (x_train / 127.5) - 1.0

    return x_train


# def build_generator():
    """Build the generator network that maps latent vectors to RGB images.

    Returns:
        A compiled-ready Keras Sequential generator model.
    """
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
    model.add(ks.layers.Dropout(0.3))

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
    model.add(ks.layers.Dropout(0.3))

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
    model.add(ks.layers.Dropout(0.3))

    # Final RGB image output: 32x32x3
    model.add(ks.layers.Conv2DTranspose(
        3,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="tanh"
    ))

    return model

def build_generator():
    """Build a more stable generator using UpSampling2D + Conv2D."""
    model = ks.Sequential(name="Generator")

    model.add(ks.layers.Input(shape=(LATENT_DIM,)))

    model.add(ks.layers.Dense(4 * 4 * 512, use_bias=False))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))
    model.add(ks.layers.Reshape((4, 4, 512)))

    # 4x4 -> 8x8
    model.add(ks.layers.UpSampling2D())
    model.add(ks.layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    # 8x8 -> 16x16
    model.add(ks.layers.UpSampling2D())
    model.add(ks.layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    # 16x16 -> 32x32
    model.add(ks.layers.UpSampling2D())
    model.add(ks.layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    model.add(ks.layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh"))

    return model

def build_discriminator():
    """Build the discriminator network that classifies real vs fake images.

    Returns:
        A compiled-ready Keras Sequential discriminator model.
    """
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


def save_generated_images(generator, epoch_label, filename, noise):
    """Render and save a 4x4 grid of generated samples.

    Args:
        generator: Trained or partially trained generator model.
        epoch_label: Text label used in the figure title.
        filename: Output image file name stored in OUTPUT_DIR.
        noise: Latent vectors used to generate the image grid.
    """
    rows, cols = 4, 4

    generated_images = generator.predict(noise, verbose=0)

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

def create_training_gif(
    image_folder=OUTPUT_DIR,
    output_filename="gan_training_progress.gif",
    duration=400
):
    """Create an animated GIF from saved epoch image grids.

    Args:
        image_folder: Directory containing generated_epoch_*.jpg files.
        output_filename: Name of the GIF file to write.
        duration: Frame display duration in milliseconds.
    """

    image_paths = glob.glob(os.path.join(image_folder, "generated_epoch_*.jpg"))

    # Sort numerically by epoch number
    image_paths = sorted(
        image_paths,
        key=lambda path: int(
            os.path.basename(path)
            .replace("generated_epoch_", "")
            .replace(".jpg", "")
        )
    )

    if not image_paths:
        print("No generated epoch images found.")
        return

    frames = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        frames.append(img)

    output_path = os.path.join(image_folder, output_filename)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    print(f"Saved training GIF: {output_path}")


def train():
    """Train the GAN and save generated outputs during and after training.

    This function trains both networks, logs progress, writes generated image
    snapshots, saves a loss curve, and builds a GIF from saved snapshots.
    """
    x_train = load_cifar10_data()

    start_time = time()

    generator = build_generator()
    print("--------------Generator Summary--------------")
    generator.summary()

    discriminator = build_discriminator()
    print("--------------Discriminator Summary--------------")
    discriminator.summary()

    # Discriminator learns slightly faster
    discriminator.trainable = True
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=ks.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"]
    )

    # Freeze discriminator only inside combined GAN
    discriminator.trainable = False

    z = ks.layers.Input(shape=(LATENT_DIM,))
    generated_image = generator(z)
    validity = discriminator(generated_image)

    combined_network = ks.Model(z, validity, name="Combined_GAN")

    # Generator learns slightly slower to reduce mode collapse
    combined_network.compile(
        loss="binary_crossentropy",
        optimizer=ks.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
    )

    discriminator.trainable = True

    d_losses = []
    g_losses = []

    # Fixed noise lets you compare the same latent inputs over time
    fixed_noise = np.random.normal(0, 1, (16, LATENT_DIM))

    for epoch in range(1, EPOCHS + 1):

        # -------------------------
        # Train Discriminator
        # -------------------------
        discriminator.trainable = True

        for _ in range(3):
            idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
            real_images = x_train[idx]

            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
            fake_images = generator.predict(noise, verbose=0)

            # Add small input noise to stabilize discriminator training
            real_images_noisy = real_images + 0.02 * np.random.normal(0, 1, real_images.shape)
            fake_images_noisy = fake_images + 0.02 * np.random.normal(0, 1, fake_images.shape)

            real_images_noisy = np.clip(real_images_noisy, -1, 1)
            fake_images_noisy = np.clip(fake_images_noisy, -1, 1)

            # Smooth only real labels; keep fake labels exactly zero
            valid = np.ones((BATCH_SIZE, 1)) * 0.9
            fake = np.zeros((BATCH_SIZE, 1))

            disc_loss_real = discriminator.train_on_batch(real_images_noisy, valid)
            disc_loss_fake = discriminator.train_on_batch(fake_images_noisy, fake)
            d_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

        # -------------------------
        # Train Generator
        # -------------------------
        discriminator.trainable = False

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

        # Slightly smoothed generator target
        misleading_targets = np.ones((BATCH_SIZE, 1)) * 0.9

        g_loss = combined_network.train_on_batch(noise, misleading_targets)

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        if epoch == 1:
            save_generated_images(
                generator,
                "Epoch 1",
                "generated_epoch_1.jpg",
                fixed_noise
            )

        if epoch % DISPLAY_INTERVAL == 0:
            print(
                f"Epoch {epoch}/{EPOCHS}\t"
                f"| D Loss: {d_loss[0]:.4f}\t"
                f"| D Accuracy: {d_loss[1] * 100:.2f}%\t"
                f"| G Loss: {g_loss:.4f}"
            )

            save_generated_images(
                generator,
                f"Epoch {epoch}",
                f"generated_epoch_{epoch}.jpg",
                fixed_noise
            )

        if epoch == EPOCHS:
            save_generated_images(
                generator,
                f"Final Epoch {EPOCHS}",
                "generated_final_epoch.jpg",
                fixed_noise
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

    loss_path = os.path.join(OUTPUT_DIR, "gan_loss_curve.jpg")
    plt.savefig(loss_path, dpi=150, format="jpg")
    plt.show(block=False)
    plt.close('all')

    print(f"Saved loss curve: {loss_path}")

    create_training_gif()

    hours, rem = divmod(time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

if __name__ == "__main__":
    train()
