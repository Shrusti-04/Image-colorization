# Image Colorization using GANs and Autoencoders

This project implements image colorization using two approaches: GANs (Generative Adversarial Networks) and Autoencoders. The system is designed to automatically colorize grayscale landscape images using deep learning techniques.

## Features

- Two colorization approaches:
  - GAN-based colorization
  - Autoencoder-based colorization
- Multiple model implementations
- Training progress visualization
- Pre-trained models included

## Project Files

### Core Models
- `autoencoder.py` - Autoencoder implementation
- `landscape_colorization_gan_v2.py` - Advanced GAN implementation
- `testing-autoencoder.py` - Autoencoder testing script

### Pre-trained Models
- `colorizer_model.h5` - Base colorization model
- `colorization_generator.h5` - Specialized generator
- `generator_model.h5` - Main generator model
- `generator_epoch_1.h5`, `generator_epoch_2.h5` - Generator checkpoints
- `discriminator_epoch_1.h5`, `discriminator_epoch_2.h5` - Discriminator checkpoints

### Results
- `Colorization_Results/` - Output directory
- `test_results_epoch_1.png` to `test_results_epoch_10.png` - Training progress
- `training_metrics.png` - Performance visualization

### Data
- `landscape Images/` - Training dataset directory

## Model Details

### GAN Architecture
- Generator: U-Net style with skip connections
- Discriminator: PatchGAN architecture
- Training parameters:
  - Adam optimizer (β1=0.5, β2=0.999)
  - Learning rate: 2e-4
  - Batch size: 16
  - Image size: 256x256

### Autoencoder Architecture
- Encoder: Convolutional downsampling
- Decoder: Transposed convolutions
- Training parameters:
  - Adam optimizer
  - Learning rate: 1e-3
  - Batch size: 32
  - Image size: 256x256

## Usage

```python
# Train GAN model
python landscape_colorization_gan_v2.py

# Train Autoencoder
python autoencoder.py

# Test Autoencoder
python testing-autoencoder.py
```

## Results

- GAN: Produces realistic and vibrant colors
- Autoencoder: Provides stable and efficient colorization
- Training progress visible in test_results_epoch_*.png
- Performance metrics in training_metrics.png

## System Requirements

- 16GB RAM minimum
- Modern multi-core CPU

## Future Improvements

1. Attention mechanisms
2. Higher resolution support
3. Real-time colorization
4. Web interface
5. Hybrid GAN-Autoencoder model

