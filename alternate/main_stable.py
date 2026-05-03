"""
Stable TensorFlow 2 / Keras WGAN-GP for CIFAR-10

Saves:
    generated_epoch_1.jpg
    generated_epoch_*.jpg
    generated_final_epoch.jpg
    gan_training_progress.gif
"""

import os
import glob
import warnings
from time import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as Python or NumPy boolean.*",
    category=Warning,
)

import tensorflow as tf
import keras as ks

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Parameters
# -----------------------------
IMAGE_SHAPE = (32, 32, 3)
LATENT_DIM = 100
CLASS_ID = 8               # 8 = ship, 6 = frog, 0 = airplane
USE_SINGLE_CLASS = True    # Set False to train on all CIFAR-10 classes

EPOCHS = 15000
BATCH_SIZE = 64
DISPLAY_INTERVAL = 250
OUTPUT_DIR = "gan_outputs"

CRITIC_UPDATES = 5
GP_WEIGHT = 10.0
GENERATOR_LR = 0.0001
CRITIC_LR = 0.0001

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_cifar10_data():
    """Load CIFAR-10, optionally selecting one class, and normalize to [-1, 1]."""
    (x_train, y_train), (_, _) = ks.datasets.cifar10.load_data()

    if USE_SINGLE_CLASS:
        x_train = x_train[y_train.flatten() == CLASS_ID]

    x_train = x_train.astype("float32")
    x_train = (x_train / 127.5) - 1.0

    return x_train


def build_generator():
    """DCGAN-style generator using UpSampling2D + Conv2D to reduce checkerboard artifacts."""
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


def build_critic():
    """
    WGAN critic.

    Unlike a normal discriminator, the critic has no sigmoid output.
    It outputs a raw score instead of a probability.
    """
    model = ks.Sequential(name="Critic")

    model.add(ks.layers.Input(shape=IMAGE_SHAPE))

    model.add(ks.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    model.add(ks.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    model.add(ks.layers.Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    model.add(ks.layers.Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.LeakyReLU(negative_slope=0.2))

    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(1))

    return model


def gradient_penalty(critic, real_images, fake_images):
    """Compute WGAN-GP gradient penalty."""
    batch_size = tf.shape(real_images)[0]

    alpha = tf.random.uniform(
        shape=[batch_size, 1, 1, 1],
        minval=0.0,
        maxval=1.0
    )

    interpolated = alpha * real_images + (1.0 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        prediction = critic(interpolated, training=True)

    gradients = tape.gradient(prediction, interpolated)
    gradients = tf.reshape(gradients, [batch_size, -1])
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-12)

    penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)

    return penalty


def save_generated_images(generator, epoch_label, filename, noise):
    """Generate a 4x4 image grid and save it as JPG."""
    generated_images = generator(noise, training=False).numpy()

    generated_images = 0.5 * generated_images + 0.5
    generated_images = np.clip(generated_images, 0, 1)

    rows, cols = 4, 4
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
    plt.savefig(save_path, dpi=150, format="jpg")
    plt.show(block=False)
    plt.close("all")

    print(f"Saved image grid: {save_path}")


def create_training_gif(
    image_folder=OUTPUT_DIR,
    output_filename="gan_training_progress.gif",
    duration=400
):
    """Create an animated GIF from generated_epoch_*.jpg images."""
    image_paths = glob.glob(os.path.join(image_folder, "generated_epoch_*.jpg"))

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

    frames = [Image.open(path).convert("RGB") for path in image_paths]

    output_path = os.path.join(image_folder, output_filename)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    print(f"Saved training GIF: {output_path}")


def train():
    """Train WGAN-GP on CIFAR-10."""
    x_train = load_cifar10_data()

    dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(buffer_size=x_train.shape[0])
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    generator = build_generator()
    critic = build_critic()

    print("--------------Generator Summary--------------")
    generator.summary()

    print("--------------Critic Summary--------------")
    critic.summary()

    generator_optimizer = ks.optimizers.Adam(
        learning_rate=GENERATOR_LR,
        beta_1=0.0,
        beta_2=0.9
    )

    critic_optimizer = ks.optimizers.Adam(
        learning_rate=CRITIC_LR,
        beta_1=0.0,
        beta_2=0.9
    )

    fixed_noise = tf.random.normal((16, LATENT_DIM))

    g_losses = []
    c_losses = []

    start_time = time()
    step = 0

    save_generated_images(
        generator,
        "Epoch 1",
        "stable_generated_epoch_1.jpg",
        fixed_noise
    )

    for epoch in range(1, EPOCHS + 1):
        real_images = next(iter(dataset))

        # -----------------------------
        # Train critic multiple times
        # -----------------------------
        for _ in range(CRITIC_UPDATES):
            real_images = next(iter(dataset))
            noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))

            with tf.GradientTape() as critic_tape:
                fake_images = generator(noise, training=True)

                real_score = critic(real_images, training=True)
                fake_score = critic(fake_images, training=True)

                gp = gradient_penalty(critic, real_images, fake_images)

                critic_loss = (
                    tf.reduce_mean(fake_score)
                    - tf.reduce_mean(real_score)
                    + GP_WEIGHT * gp
                )

            critic_grads = critic_tape.gradient(
                critic_loss,
                critic.trainable_variables
            )

            critic_optimizer.apply_gradients(
                zip(critic_grads, critic.trainable_variables)
            )

        # -----------------------------
        # Train generator once
        # -----------------------------
        noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))

        with tf.GradientTape() as gen_tape:
            fake_images = generator(noise, training=True)
            fake_score = critic(fake_images, training=True)
            generator_loss = -tf.reduce_mean(fake_score)

        gen_grads = gen_tape.gradient(
            generator_loss,
            generator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(gen_grads, generator.trainable_variables)
        )

        g_losses.append(float(generator_loss))
        c_losses.append(float(critic_loss))

        step += 1

        if epoch % DISPLAY_INTERVAL == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| Critic Loss: {float(critic_loss):.4f} "
                f"| Generator Loss: {float(generator_loss):.4f}"
            )

            save_generated_images(
                generator,
                f"Epoch {epoch}",
                f"stable_generated_epoch_{epoch}.jpg",
                fixed_noise
            )

        if epoch == EPOCHS:
            save_generated_images(
                generator,
                f"Final Epoch {EPOCHS}",
                "stable_generated_final_epoch.jpg",
                fixed_noise
            )

    plt.figure(figsize=(8, 5))
    plt.plot(c_losses, label="Critic Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("WGAN-GP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(OUTPUT_DIR, "gan_loss_curve.jpg")
    plt.savefig(loss_path, dpi=150, format="jpg")
    plt.show(block=False)
    plt.close("all")

    print(f"Saved loss curve: {loss_path}")

    create_training_gif()

    hours, rem = divmod(time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


if __name__ == "__main__":
    train()