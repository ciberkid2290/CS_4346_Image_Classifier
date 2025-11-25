# Goal: Convert image data into feature vectors
import numpy as np

# Flattens each 2D digit image into a 1D feature vector
def extract_raw_features(image):
    return image.flatten()

# Extracts features by dividing image into a grid and checking for foreground pixels in each cell
def extract_grid_features(image, grid_width, grid_height):
    img_height, img_width = image.shape
    cell_height = img_height // grid_height
    cell_width = img_width // grid_width
    
    features = []
    for i in range(grid_height):
        for j in range(grid_width):
            # Define cell boundaries
            start_row, end_row = i * cell_height, (i + 1) * cell_height
            start_col, end_col = j * cell_width, (j + 1) * cell_width
            cell = image[start_row:end_row, start_col:end_col]
            # Feature is 1 if the sum of pixels in the cell is greater than 0, else 0
            features.append(1 if np.sum(cell) > 0 else 0)
    return np.array(features)

# Applies a feature extraction function to a list of images
def extract_features_for_dataset(images, feature_type = 'raw', **kwargs):
    feature_vectors = []
    for image in images:
        if feature_type == 'raw':
            feature_vectors.append(extract_raw_features(image))
        elif feature_type == 'grid':
            feature_vectors.append(extract_grid_features(image, kwargs['grid_width'], kwargs['grid_height']))
    
    return np.array(feature_vectors)