# Goal: Load image and label data from ASCII files
import numpy as np

def load_images(image_file, num_images, width, height):
    """
    Reads the image file and returns a numpy array of shape
    (num_images, height, width) with binary pixel values (0 or 1).
    """
    images = []

    with open(image_file, 'r') as f:
        lines = f.readlines()

    # Robust check: File must contain exactly num_images * height lines
    expected_lines = num_images * height
    if len(lines) < expected_lines:
        raise ValueError(f"Expected {expected_lines} lines in image file, got {len(lines)}")
    
    # Index to track current line in file
    idx = 0
    for i in range(num_images):
        img_rows = []
        for j in range(height):
            line = lines[idx].rstrip('\n')
            idx += 1

            # Pad or trim logic from newer models
            if len(line) < width:
                line = line + ' ' * (width - len(line))
            elif len(line) > width:
                line = line[:width]
            
            # Convert: Space -> 0, Anything else -> 1
            row = [0 if ch == ' ' else 1 for ch in line]
            img_rows.append(row)
        
        images.append(img_rows)
    
    return np.array(images, dtype=np.int8)

def load_labels(label_file, num_images = None):
    """
    Reads the label file and returns a numpy array of integers.
    If num_images is None, reads all labels in the file.
    """
    labels = []
    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            # If a limit is set, stop when we reach it
            if num_images is not None and i >= num_images:
                break

            label_str = line.strip()
            if label_str == '':
                continue

            labels.append(int(label_str))
    
    # Validation: If we expected num_images but got a different count, raise error
    if num_images is not None and len(labels) < num_images:
        raise ValueError(f"Expected {num_images} labels, got {len(labels)}")
    
    return np.array(labels, dtype=np.int32)

def load_data(image_file, label_file, width, height):
    """
    Main entry point. Leads the loading process using the functions above.
    Returns:
        images: np.ndarray of shape (N, height, width)
        labels: np.ndarray of shape (N,)
    """
    # Load labels first to determine number of images
    labels = load_labels(label_file, num_images = None)
    num_images = len(labels)

    # Load images using count determined from labels
    images = load_images(image_file, num_images, width, height)

    return images, labels