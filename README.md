# Metasurface Generation via ADAM Optimization

**Author:** Ivan Ling  
**Affiliation:** University of Southampton, 2025

## 📖 Overview

This project implements an inverse design framework for generating metasurfaces using PyTorch. Instead of traditional heuristic methods, we utilize the **ADAM optimizer** (gradient descent) to iteratively adjust the parameters of a surface distribution to match a target shadow pattern.

The core idea is to represent the metasurface as a collection of differentiable geometric primitives (soft circles) and minimize the Mean Squared Error (MSE) between the generated shadow and a target image.

## 🚀 Key Features

- **Differentiable Rendering:** Uses PyTorch to create a differentiable pipeline from surface parameters to shadow generation.
- **Gradient-Based Optimization:** Leverages ADAM to converge on optimal surface parameters faster than brute-force methods.
- **Feature Reduction:** Uses a Fourier-series inspired approach (sum of circles) to reduce the dimensionality of the problem.
- **GPU Acceleration:** Fully compatible with CUDA for faster training on NVIDIA GPUs.

## 🛠️ Requirements

To run this project, you need Python installed along with the following libraries:

```bash
pip install numpy matplotlib opencv-python torch
