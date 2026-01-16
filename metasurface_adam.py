"""
Metasurface Generation with ADAM Optimization
Author: Ivan Ling
University of Southampton, 2025

Description:
This script generates a metasurface distribution based on a 2D Fourier Series representation.
It uses PyTorch and the ADAM optimizer to reconstruct a target shadow pattern (image)
by optimizing the coefficients of the surface generation function.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F

# --- Configuration & Hyperparameters ---
IMAGE_PATH = 'target_shadow.png'  # Ensure you have a target image or modify the code to generate one
RESOLUTION = 256                  # Resolution of the grid (256x256)
NUM_CIRCLES = 50                  # Number of Fourier terms/circles
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
RAD_MIN = 0.02
RAD_SCALE = 0.1

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_mask(X, Y, coeffs, num_circles, res, rad_min, rad_scale):
    """
    Generates a mask based on 2D Fourier-like circular expansion.
    """
    mask = torch.zeros((res, res), device=device)
    
    # We interpret coeffs as parameters controlling circle positions/sizes
    # This loop is a placeholder logic based on the notebook structure.
    # The actual mathematical definition depends on how coeffs map to circle properties.
    # Below is a standard differentiable rendering approximation.
    
    for i in range(num_circles):
        # Extract parameters for the i-th circle from coeffs
        # Assuming coeffs has shape (num_circles, 3) -> x, y, radius_factor
        cx = coeffs[i, 0]
        cy = coeffs[i, 1]
        r_factor = coeffs[i, 2]
        
        radius = rad_min + torch.sigmoid(r_factor) * rad_scale
        
        # Soft mask generation (differentiable)
        dist_sq = (X - cx)**2 + (Y - cy)**2
        # Sigmoid used to create a soft edge for differentiability
        circle = torch.sigmoid((radius**2 - dist_sq) * 1000) 
        
        mask = torch.max(mask, circle)
        
    return mask

def generate_shadow(mask):
    """
    Simulates the shadow cast by the mask. 
    (Simple identity mapping or specific diffraction logic can be placed here)
    """
    # For now, assuming direct projection/shadow
    return mask

def main():
    # 1. Setup Coordinate Grid
    x = torch.linspace(-1, 1, RESOLUTION, device=device)
    y = torch.linspace(-1, 1, RESOLUTION, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # 2. Load Target Image
    # If no image exists, we create a dummy target (a simple circle) for demonstration
    try:
        target_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if target_img is None: raise FileNotFoundError
        target_img = cv2.resize(target_img, (RESOLUTION, RESOLUTION))
        target_img = target_img.astype(np.float32) / 255.0
    except (FileNotFoundError, Exception):
        print("Target image not found. generating a synthetic target circle.")
        target_img = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)
        cv2.circle(target_img, (RESOLUTION//2, RESOLUTION//2), 50, 1, -1)

    target_tensor = torch.tensor(target_img, device=device)

    # 3. Initialize Learnable Coefficients
    # Shape: [num_circles, 3] for (x_pos, y_pos, radius_param)
    coeffs = torch.randn(NUM_CIRCLES, 3, device=device, requires_grad=True)

    # 4. Optimizer Setup
    optimizer = optim.Adam([coeffs], lr=LEARNING_RATE)
    loss_history = []

    print("Starting Optimization...")

    # 5. Training Loop
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        # Forward pass
        approx_mask = generate_mask(X, Y, coeffs, NUM_CIRCLES, RESOLUTION, RAD_MIN, RAD_SCALE)
        approx_shadow = generate_shadow(approx_mask)

        # Compute Loss (MSE)
        loss = F.mse_loss(approx_shadow, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')

    # 6. Visualization
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')

    # Plot Target
    plt.subplot(1, 3, 2)
    plt.imshow(target_tensor.cpu().numpy(), cmap='gray')
    plt.title('Target Shadow')

    # Plot Result
    with torch.no_grad():
        final_mask = generate_mask(X, Y, coeffs, NUM_CIRCLES, RESOLUTION, RAD_MIN, RAD_SCALE)
        final_shadow = generate_shadow(final_mask)
        
    plt.subplot(1, 3, 3)
    plt.imshow(final_shadow.cpu().numpy(), cmap='gray')
    plt.title('Generated Metasurface')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()