import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load grayscale image
image_path = (r'C:\Users\ADMIN\OneDrive\Documents\Documents\RUPP\code\Python\DenoisingImages\Wallpaper.png')
#image_path = (r"C:\Users\ADMIN\OneDrive\Documents\Documents\RUPP\code\Python\DenoisingImages\Peacce.jpg")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise (for testing)
mean, std_dev = 0, np.std(img) * 0.5
noise = np.random.normal(mean, std_dev, img.shape)
noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

# Perform SVD
U, S, Vt = np.linalg.svd(noisy_img, full_matrices=False)

#Choose k, number of singular values to retain & Try reducing k to 350 (or lower)
k_opt = 300  

# Keep only top k singular values
S_k = np.zeros_like(S)
S_k[:k_opt] = S[:k_opt]

# Reconstruct the image
denoised_img = np.dot(U, np.dot(np.diag(S_k), Vt))
denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

# Apply bilateral filter for noise reduction while keeping edges
final_denoised_img = cv2.bilateralFilter(denoised_img, d=9, sigmaColor=50, sigmaSpace=50)

# Show images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(final_denoised_img, cmap='gray')
plt.title(f"Denoised Image (k={k_opt})")
plt.axis('off')

plt.tight_layout()
plt.show()
