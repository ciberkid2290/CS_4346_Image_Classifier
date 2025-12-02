# Features extraction functions for digit and face datasets
# Each image is a 2D numpy array of 0/1 values; 0 = background, 1 = foreground

# We provide specific feature extractors for digits and faces:
# - digit_features_raw: raw pixel values for digit images
# - digit_features_blocks: block-based features for digit images
# - face_features_raw: raw pixel values for face images
# - face_features_regions: region-based features for face images

# Every function will return a 1D numpy array of features

import numpy as np

# --------------------------------
# Helper functions
# --------------------------------
# Common helper functions for feature extraction
def flatten_image(image: np.ndarray) -> np.ndarray:
    """Flatten a 2D image into a 1D feature array."""
    return image.ravel().astype(np.int8)

def block_density_features(image: np.ndarray, num_rows: int,
                           num_cols: int, threshold: float = 0.2, ) -> np.ndarray:
    """
    Splits the image into a (num_rows x num_cols) block grid and computes a binary feature
    for each block based on its pixel density

    For each block:
        - Computes the fraction of foreground pixels (value 1)
        - If the density is greater than the threshold, the feature is 1, else 0

    This aims to create a lower-dimensional representation that captures
    the general shape and structure of the image content.

    Returns:
        A 1D numpy array of length (num_rows * num_cols) with binary features.
    """
    features = []
    # Split image into horizontal strips
    for row_strip in np.array_split(image, num_rows, axis = 0):
        # Split each horizontal strip into vertical blocks
        for block in np.array_split(row_strip, num_cols, axis = 1):
            if block.size == 0: # Handle edge case of empty blocks
                features.append(0)
                continue

            density = np.mean(block) # Compute density of foreground pixels
            features.append(1 if density > threshold else 0)
    return np.array(features, dtype = np.int8)

# Helper to compute density feature for a given region
def get_region_density_features(region: np.ndarray, threshold: float) -> int:
    if region.size == 0:
        return 0
    density = np.mean(region)
    return 1 if density > threshold else 0


# --------------------------------
# Digit features
# --------------------------------
def digit_features_raw(image: np.ndarray) -> np.ndarray:
    """
    Extract raw pixel features from a digit image.
    Input: 2D Array (e.g., 28x28) of 0/1 values
    Output: 1D Array of length H * W (784 features for 28x28 images)
    """
    return flatten_image(image)

def digit_features_blocks(image: np.ndarray) -> np.ndarray:
    """
    Feature set 2 for digits: Block-based features.
    
    Divides the 28x28 image into a 4x4 grid (16 blocks total) and marks each block
    as 1 if the density of foreground pixels exceeds a threshold, else 0.
    """
    return block_density_features(image, num_rows = 4, num_cols = 4, threshold = 0.2)

# --------------------------------
# Face features
# --------------------------------
def face_features_raw(image: np.ndarray) -> np.ndarray:
    """
    Feature set 1 for faces: Raw pixel (flattened).

    Input: Image: 2D Array (e.g., 70x60) of 0/1 values
    Output: 1D Array of length H * W (4200 features for 70x60 images)
    """
    return flatten_image(image)

def face_features_regions(image: np.ndarray) -> np.ndarray:
    """
    Feature set 2 for faces: Large region density features.

    This creates 4 features based on foreground pixel density in 4 large,
    overlapping regions of the image:
    - Upper half
    - Lower half
    - Left half
    - Right half

    Each region's feature is set to 1 if its density exceeds a threshold, else 0.
    """
    H, W = image.shape
    mid_row = H // 2
    mid_col = W // 2
    threshold = 0.15

    # Define the regions using array slicing
    reigons = [
        image[0:mid_row, :],     # Upper half
        image[mid_row:H, :],     # Lower half
        image[:, 0:mid_col],     # Left half
        image[:, mid_col:W],     # Right half
    ]

    # Calculate density features for each region
    features = [get_region_density_features(region, threshold) for region in reigons]
    return np.array(features, dtype = np.int8)

