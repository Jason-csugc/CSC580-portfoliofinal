# CSC580-portfoliofinal

CSC580 Portfolio Final Project

## Product

### Overview

This repository contains a TensorFlow and Keras GAN workflow focused on
CIFAR-10 image generation. The primary training script learns to generate
images for a selected CIFAR-10 class (default: class 8, ship), saves
intermediate image grids during training, exports a final image grid, plots
generator/discriminator loss curves, and builds an animated training-progress
GIF.

The project also includes alternate experiment scripts in the alternate folder
for reduced and stable variants, plus historical prototypes.

## Features

- load_cifar10_data() in main.py: loads CIFAR-10 data, optionally filters a
	single class, and normalizes images to [-1, 1]
- build_generator() in main.py: defines an upsampling-based generator for
	32x32 RGB image synthesis
- build_discriminator() in main.py: defines a CNN discriminator for real/fake
	scoring
- save_generated_images() in main.py: saves 4x4 generated image grids at
	selected epochs
- create_training_gif() in main.py: creates gan_training_progress.gif from
	generated_epoch_*.jpg snapshots
- train() in main.py: orchestrates adversarial training, periodic logging,
	artifact export, and training-time reporting
- train() in alternate/main_stable.py: runs a more stable WGAN-GP style
	training loop with critic updates and gradient penalty
- train() in alternate/main_reduced.py: runs a reduced/shorter baseline setup
	for experimentation
- alternate/main.py, alternate/main_1.py, alternate/main_mnist.py: additional
	archived experimental variants for comparison and iteration history

## Getting Started

1. Create and activate a virtual environment:

```bash
python3 -m venv mod8pf
source mod8pf/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main portfolio training configuration:

```bash
python main.py
```

4. Optional: run alternate training variants:

```bash
python alternate/main_stable.py
python alternate/main_reduced.py
python alternate/main.py
```

## Notes

- Designed for coursework use in CSC580 portfolio/final deliverables.
- Default configuration uses CIFAR-10 class 8 (ship) with fixed random seeds
	for reproducibility.
- TensorFlow logging is reduced via TF_CPP_MIN_LOG_LEVEL and oneDNN
	optimizations are disabled for more consistent behavior across environments.
- The scripts currently save image artifacts and plots; they do not export a
	standalone .keras generator checkpoint by default.
- Long training schedules are configured (up to 15000 epochs), so GPU
	acceleration is strongly recommended.
- Existing artifact subsets are organized under gan_outputs/final,
	gan_outputs/reduced, gan_outputs/stable, and gan_outputs/transpose.

## Outputs

1. Console output:
- Model summaries for generator/discriminator (or critic)
- Epoch progress logs with discriminator/critic and generator losses
- Periodic discriminator accuracy reporting (baseline scripts)
- Final training time summary

<img width="612" height="611" alt="image" src="https://github.com/user-attachments/assets/3aae3acf-ad18-4b29-a938-47db18a3d344" />

<img width="613" height="612" alt="image" src="https://github.com/user-attachments/assets/c1085824-72cd-40b5-a17f-a6d10a88464a" />

<img width="612" height="608" alt="image" src="https://github.com/user-attachments/assets/c6e3b59f-7dc6-4658-928d-a9283f8435a6" />

<img width="609" height="382" alt="image" src="https://github.com/user-attachments/assets/65a7659b-bbac-4647-b49e-0a55ccd3a4b8" />


2. Artifacts:
- Generated image snapshots such as gan_outputs/generated_epoch_250.jpg
- Final generated sample grid such as gan_outputs/generated_final_epoch.jpg
- Loss curve plot: gan_outputs/gan_loss_curve.jpg
- Animated progression GIF: gan_outputs/gan_training_progress.gif

Example artifact previews from tracked outputs:

![Final generated grid](gan_outputs/final/generated_final_epoch.jpg)

![Training loss curve](gan_outputs/final/gan_loss_curve.jpg)

![Stable variant final grid](gan_outputs/stable/stable_generated_final_epoch.jpg)

## Additional Links

- [Code](https://github.com/Jason-csugc/CSC580-portfoliofinal)
- [Issues](https://github.com/Jason-csugc/CSC580-portfoliofinal/issues)
- [Pull requests](https://github.com/Jason-csugc/CSC580-portfoliofinal/pulls)
- [Actions](https://github.com/Jason-csugc/CSC580-portfoliofinal/actions)
- [Projects](https://github.com/Jason-csugc/CSC580-portfoliofinal/projects)
- [Security and quality](https://github.com/Jason-csugc/CSC580-portfoliofinal/security)
