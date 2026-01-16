# Metasurface Generation via ADAM Optimization

**Author:** Dr. Ivan Ling  
**Affiliation:** University of Southampton, 2025

## 📖 Overview

This project implements an inverse design framework for generating metasurfaces using PyTorch. Instead of traditional heuristic methods, we utilize the **ADAM optimizer** (gradient descent) to iteratively adjust the parameters of a surface distribution to match a target shadow pattern.

The core idea is to represent the metasurface as a collection of differentiable geometric primitives (soft circles) and minimize the Mean Squared Error (MSE) between the generated shadow and a target image.

## 🧠 Motivations & Physics

### The Inverse Design Challenge
Designing metasurfaces typically involves "inverse design": we know the desired optical response (e.g., a specific hologram or shadow pattern in the far-field), but finding the physical surface geometry that produces it is computationally intensive. Traditional topology optimization treats every point on the surface as a variable, resulting in a search space with thousands of degrees of freedom.

### Physics of Wavefront Manipulation
Metasurfaces work by introducing local phase discontinuities to an incident wavefront. As light passes through (or reflects off) the diverse height distributions of the surface, it accumulates different amounts of phase delay.

According to the Huygens-Fresnel principle, every point on this wavefront acts as a source of secondary spherical wavelets. These wavelets interfere constructively and destructively as they propagate. The resulting pattern observed at a distance (the far-field) is essentially the Fourier transform of this complex field at the surface. By carefully engineering the surface geometry, we can dictate exactly how these waves interfere to form specific images or focal points.

### Why Feature Reduction?
This project moves away from pixel-by-pixel optimization in favor of a parametric approach (using a sum of Fourier-like circular functions). This is crucial for two reasons:

- Enabling Machine Learning: Standard high-resolution metasurfaces (e.g., 256x256 grids) present a "curse of dimensionality" for neural networks. By reducing the surface representation to a small set of coefficients (e.g., 50 circle parameters), we compress the problem significantly. This allows lightweight Machine Learning algorithms to learn the mapping between target images and surface parameters, enabling rapid generation of masks without running slow iterative optimizations every time.

- Simulation & Fabrication Viability: The generated masks act as phase-plates. Because they are defined by smooth geometric functions rather than random noisy pixels, the resulting structures are more continuous. This makes them highly suitable for:

- Phase Simulations: Calculating the accumulated phase for transmission or reflection profiles.

- Far-Field Propagation: Using Fast Fourier Transforms (FFT) to predict the diffraction pattern at a distance.

## 🚀 Key Features

- **Differentiable Rendering:** Uses PyTorch to create a differentiable pipeline from surface parameters to shadow generation.
- **Gradient-Based Optimization:** Leverages ADAM to converge on optimal surface parameters faster than brute-force methods.
- **Feature Reduction:** Uses a Fourier-series inspired approach (sum of circles) to reduce the dimensionality of the problem.
- **GPU Acceleration:** Fully compatible with CUDA for faster training on NVIDIA GPUs.

## 🛠️ Requirements

To run this project, you need Python installed along with the following libraries:

```bash
pip install numpy matplotlib opencv-python torch
```
Alternatively, you can just use the requirements.txt file provided
```bash
pip install -r requirements.txt
```

