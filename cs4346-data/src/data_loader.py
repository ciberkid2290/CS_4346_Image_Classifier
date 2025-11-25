# Goal: Load image and label data from binary files
import numpy as np

# Constants for image dimensions
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
FACE_WIDTH = 60
FACE_HEIGHT = 70

# Loads image and label data from binary files
def load_data(image_file, label_file, width, height):
    #Read labels
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
        
    #Read images
    with open(image_file, 'r') as f:
        lines = f.readlines()
    
    images = []
    num_images = len(labels) # Assuming one label per image
    # Process each image
    for i in range(num_images):
        start = i * height
        end = start + height
        image_lines = lines[start:end]

        # Convert lines to a 2D numpy array (0 for space, 1 for #/+)
        image = []
        for line in image_lines:
            row = [1 if char in ['#', '+'] else 0 for char in line.rstrip('\n')]
            row.extend([0] * (width - len(row)))  # Pad row if necessary to ensure rows are consistent
            image.append(row)
        images.append(np.array(image))
        
    return np.array(images), np.array(labels)