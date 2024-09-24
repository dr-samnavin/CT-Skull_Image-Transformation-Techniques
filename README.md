# Step 1: Install necessary libraries
!pip install opencv-python-headless
!pip install pywavelets  # For Haar transform

# Step 2: Import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.fft import fft2, ifft2
from pywt import wavedec2, waverec2  # PyWavelets (wavelet transforms)
from scipy.linalg import hadamard
from sklearn.decomposition import PCA
import os  # For checking file existence

# Helper Function for Normalizing Transform Results
def normalize_image(image):
    image = np.abs(image)  # Absolute value for frequency domain transforms
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255  # Normalize to [0, 255]
    return image.astype(np.uint8)

# Step 3: Load the CT image from disk if it exists
def load_ct_image():
    filename = 'SKULL_CT.jpg' #source-uman_Skull_2 Free DICOM file by Medimodel team (https://medimodel.com/sample-dicom-files/human_skull_2_dicom_file/)
    
    if os.path.exists(filename):
        print(f"File '{filename}' found, loading the image.")
        # Read the image in grayscale using OpenCV
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        raise FileNotFoundError(f"File '{filename}' not found. Please ensure it is uploaded or in the environment.")
    
    # Resize image to manageable dimensions
    image = cv2.resize(image, (256, 256))  # Resize to 256x256 for consistency
    return image

# Step 4: Display two images side by side (Original and Transformed)
def display_images_side_by_side(original, transformed, title1="Original", title2="Transformed"):
    plt.figure(figsize=(10, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    # Display transformed image
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(transformed) + 1), cmap='gray')  # Log scale for better visualization
    plt.title(title2)
    plt.axis('off')

    # Show both images
    plt.show()

  # Step 5: Perform DCT (Discrete Cosine Transform)
def perform_dct(image):
    # Apply 2D DCT
    dct_transformed = dct(dct(image.T, norm='ortho').T, norm='ortho')
    return dct_transformed

# Step 6: Perform DFT (Discrete Fourier Transform)
def perform_dft(image):
    # Apply 2D DFT
    dft_transformed = np.fft.fftshift(fft2(image))
    return dft_transformed

# Step 7: Perform Haar Wavelet Transform
def perform_haar(image):
    # Apply 2D Haar Wavelet Transform (1-level decomposition)
    coeffs = wavedec2(image, 'haar', level=1)
    return coeffs

# Step 8: Perform Hadamard Transform
def pad_to_power_of_2(image):
    # Pad image to the nearest power of 2
    rows, cols = image.shape
    new_size = 2 ** int(np.ceil(np.log2(max(rows, cols))))
    
    # Create a new square image with padding (if necessary)
    padded_image = np.zeros((new_size, new_size))
    padded_image[:rows, :cols] = image  # Place the original image in the top-left corner
    return padded_image

def perform_hadamard(image):
    # Pad the image to the nearest power of 2
    padded_image = pad_to_power_of_2(image)
    
    # Generate Hadamard matrix for the size of the padded image
    N = hadamard(padded_image.shape[0])
    
    # Apply Hadamard transform
    hadamard_transformed = np.dot(np.dot(N, padded_image), N)
    
    # Crop back to the original image size after transform
    return hadamard_transformed[:image.shape[0], :image.shape[1]]

# Step 9: Perform KL Transform (PCA Approximation)
def perform_kl_transform(images):
    # Flatten images for PCA
    flat_images = [img.flatten() for img in images]
    
    # Apply PCA (KLT)
    pca = PCA(n_components=0.95)  # Retain 95% variance
    pca_transformed = pca.fit_transform(flat_images)
    return pca_transformed

# Step 10: Load the CT image
image = load_ct_image()

# Step 11: Apply and display DCT
dct_result = perform_dct(image)
display_images_side_by_side(image, dct_result, "Original CT Image", "DCT of CT Image")

# Step 12: Apply and display DFT
dft_result = perform_dft(image)
display_images_side_by_side(image, dft_result, "Original CT Image", "DFT of CT Image")

# Step 13: Apply and display Haar Wavelet Transform (Approximation coefficients only)
haar_result = perform_haar(image)
display_images_side_by_side(image, haar_result[0], "Original CT Image", "Haar Wavelet Transform Approximation")

# Step 14: Apply and display Hadamard Transform
hadamard_result = perform_hadamard(image)
display_images_side_by_side(image, hadamard_result, "Original CT Image", "Hadamard Transform of CT Image")

# Step 15: Perform KL Transform (PCA-based) for a single image
def perform_kl_transform_single(image):
    # Flatten the image
    flat_image = image.flatten().reshape(1, -1)
    
    # Apply PCA (KLT)
    pca = PCA(n_components=0.95)  # Retain 95% variance
    pca_transformed = pca.fit_transform(flat_image)
    
    # Reconstruct the image after PCA
    reconstructed_image = pca.inverse_transform(pca_transformed).reshape(image.shape)
    
    return reconstructed_image

# Step 15: Apply and display KL Transform (PCA-based)
kl_result = perform_kl_transform_single(image)
display_images_side_by_side(image, kl_result, "Original CT Image", "KL Transform (PCA-based) Approximation")

# Perform Inverse DCT
def perform_idct(dct_image):
    # Apply 2D iDCT
    idct_transformed = idct(idct(dct_image.T, norm='ortho').T, norm='ortho')
    return idct_transformed

# Apply iDCT and display
idct_result = perform_idct(dct_result)
display_images_side_by_side(image, normalize_image(idct_result), "Original CT Image", "Reconstructed iDCT Image")

# Perform Inverse DFT
def perform_idft(dft_image):
    # Shift back and apply iDFT
    idft_transformed = ifft2(np.fft.ifftshift(dft_image)).real  # Take the real part
    return idft_transformed

# Apply iDFT and display
idft_result = perform_idft(dft_result)
display_images_side_by_side(image, normalize_image(idft_result), "Original CT Image", "Reconstructed iDFT Image")

# Perform Inverse Hadamard Transform
def perform_ihadamard(hadamard_image):
    # Apply Hadamard transform again (same process as forward Hadamard)
    padded_image = pad_to_power_of_2(hadamard_image)
    N = hadamard(padded_image.shape[0])
    ihadamard_transformed = np.dot(np.dot(N, padded_image), N)
    return ihadamard_transformed[:image.shape[0], :image.shape[1]]  # Crop back to original size

# Apply iHadamard and display
ihadamard_result = perform_ihadamard(hadamard_result)
display_images_side_by_side(image, normalize_image(ihadamard_result), "Original CT Image", "Reconstructed iHadamard Image")

