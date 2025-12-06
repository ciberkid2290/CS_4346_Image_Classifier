import time # Importing time module for performance measurement
import os   # Importing os module for file path operations
import numpy as np # Importing numpy for numerical operations

from data_loader import load_images, load_labels # Importing function to load images
from features import (              # Importing feature extraction functions
    digit_features_raw,
    digit_features_blocks,
    face_features_raw,
    face_features_regions,
)
from naive_bayes import NaiveBayesClassifier as NaiveBayes
from perceptron import PerceptronClassifier as Perceptron

# Count how many labels in a label file
def count_labels(label_file):
    count = 0
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip() != '':
                count += 1
    return count

# Dataset loading wrappers
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28  
FACE_WIDTH = 60
FACE_HEIGHT = 70

# Load the digit dataset
def load_digit_data():
    train_images_file = os.path.join("digitdata", "trainingimages")
    train_labels_file = os.path.join("digitdata", "traininglabels")
    test_images_file = os.path.join("digitdata", "testimages")
    test_labels_file = os.path.join("digitdata", "testlabels")

    train_labels = load_labels(train_labels_file)
    test_labels = load_labels(test_labels_file)
    
    train_images = load_images(train_images_file, len(train_labels), DIGIT_WIDTH, DIGIT_HEIGHT)
    test_images = load_images(test_images_file, len(test_labels), DIGIT_WIDTH, DIGIT_HEIGHT)

    return train_images, train_labels, test_images, test_labels

# Load the face dataset
def load_face_data():
    # Load face training and testing data
    train_images_file = os.path.join("facedata", "facedatatrain")
    train_labels_file = os.path.join("facedata", "facedatatrainlabels")
    test_images_file = os.path.join("facedata", "facedatatest")
    test_labels_file = os.path.join("facedata", "facedatatestlabels")

    train_labels = load_labels(train_labels_file)
    test_labels = load_labels(test_labels_file)
    
    train_images = load_images(train_images_file, len(train_labels), FACE_WIDTH, FACE_HEIGHT)
    test_images = load_images(test_images_file, len(test_labels), FACE_WIDTH, FACE_HEIGHT)

    return train_images, train_labels, test_images, test_labels


# Feature extraction
def build_feature_matrix(images, feature_function):
    # convert list of 2D images to feature matrix
    feature_list = [feature_function(img) for img in images]
    return np.array(feature_list, dtype=np.int8)

# Experiment runner (single experiment)
def run_single_experiment(
        model_class,
        model_name,
        dataset_name,
        num_classes,
        train_images,
        train_labels,
        test_images,
        test_labels,
        feature_func,
        feature_name,
        fractions,
        num_runs,
        model_kwargs = None,
):
    # Run a single, defined experiment configuration
    if model_kwargs is None:
        model_kwargs = {}
    
    # Number of training examples
    N_train = len(train_images)
    
    # Print experiment header
    print("-" * 60)
    print(f"Running Experiment: {model_name} on {dataset_name} with {feature_name}")
    print("-" * 60)
    
    # Build feature matricies
    print("Building feature matrices...")
    X_train_full = build_feature_matrix(train_images, feature_func)
    X_test_full = build_feature_matrix(test_images, feature_func)
    y_train_full = np.array(train_labels)
    y_test = np.array(test_labels)
    print(f"Train features: {X_train_full.shape}, Test features: {X_test_full.shape}")

    # Loop over training fractions
    for frac in fractions:
        k = int(N_train * frac)
        if k < 1: k = 1
        print(f"\nTraining with {k} / {N_train} training examples ({frac:.0%})")
        accuracies, runtimes = [], []
        for run in range(num_runs):
            t0 = time.time()
            indicies = np.random.choice(N_train, size = k, replace = False)
            X_train, y_train = X_train_full[indicies], y_train_full[indicies]

            model = model_class(num_classes = num_classes, **model_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_full)
            accuracy = np.mean(y_pred == y_test)

            t1 = time.time()
            accuracies.append(accuracy)
            runtimes.append(t1 - t0)

            print(f" Run {run+1}/{num_runs}: Accuracy = {accuracy:.4f}, Time = {runtimes[-1]:.3f}s")

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            mean_time = np.mean(runtimes)

            print(
                f" Summary for {frac:.0%}: "
                f"Mean Accuracy = {mean_acc:.4f} (std: {std_acc:.4f}), "
                f"Avg Time = {mean_time:.3f}s"
            )
        print("\n")

# Main execution block

def main():
    # Main function to define and run all experiments
    np.random.seed(42) # For reproducibility

    # Load datasets
    print("Loading all datasets...")
    datasets = {
        "Digits": load_digit_data,
        "Faces": load_face_data,
    
    }
    print("...datasets loaded.\n")

    # Experiment settings
    fractions = [i / 10.0 for i in range (1, 11)]  # 10% to 100%
    num_runs = 5  # Number of runs per fraction

    # Define experiments with list of dictionaries defining each experiment
    experiments_configs = [
        # Digit - Naive Bayes - Raw Pixels
        {
            "model_class": NaiveBayes, "model_name": "NaiveBayes", "dataset_name": "Digits",
            "num_classes": 10, "feature_func": digit_features_raw, "feature_name": "RawPixels",
            "model_kwargs": {"smoothing": 1.0},
        },
        # Digit - Naive Bayes - Block Features
        {
            "model_class": NaiveBayes, "model_name": "NaiveBayes", "dataset_name": "Digits",
            "num_classes": 10, "feature_func": digit_features_blocks, "feature_name": "BlockFeatures",
            "model_kwargs": {"smoothing": 1.0},
        },
        # Face - Naive Bayes - Raw Pixels
        {
            "model_class": NaiveBayes, "model_name": "NaiveBayes", "dataset_name": "Faces",
            "num_classes": 2, "feature_func": face_features_raw, "feature_name": "RawPixels",
            "model_kwargs": {"smoothing": 1.0},
        },
        # Face - Naive Bayes - Region Features
        {
            "model_class": NaiveBayes, "model_name": "NaiveBayes", "dataset_name": "Faces",
            "num_classes": 2, "feature_func": face_features_regions, "feature_name": "BlockFeatures",
            "model_kwargs": {"smoothing": 1.0},
        },
        # Digit - Perceptron - Raw Pixels
        {
            "model_class": Perceptron, "model_name": "Perceptron", "dataset_name": "Digits",
            "num_classes": 10, "feature_func": digit_features_raw, "feature_name": "RawPixels",
            "model_kwargs": {"max_epochs": 10, "learning_rate": 1.0, "shuffle": True},
        
        },
        # Digit - Perceptron - Block Features
        {
            "model_class": Perceptron, "model_name": "Perceptron", "dataset_name": "Digits",
            "num_classes": 10, "feature_func": digit_features_blocks, "feature_name": "BlockFeatures",
            "model_kwargs": {"max_epochs": 10, "learning_rate": 1.0, "shuffle": True},
        },
        # Face - Perceptron - Raw Pixels
        {
            "model_class": Perceptron, "model_name": "Perceptron", "dataset_name": "Faces",
            "num_classes": 2, "feature_func": face_features_raw, "feature_name": "RawPixels",
            "model_kwargs": {"max_epochs": 10, "learning_rate": 1.0, "shuffle": True},
        },
        # Face - Perceptron - Region Features
        {
            "model_class": Perceptron, "model_name": "Perceptron", "dataset_name": "Faces",
            "num_classes": 2, "feature_func": face_features_regions, "feature_name": "BlockFeatures",
            "model_kwargs": {"max_epochs": 10, "learning_rate": 1.0, "shuffle": True},
        },
    ]
    # Run all experiments
    for config in experiments_configs:
        # Unpack data for current experiment
        train_img, train_lbl, test_img, test_lbl = datasets[config["dataset_name"]]()
        
        # Run the experiment with specified configuration
        run_single_experiment(
            train_images = train_img,
            train_labels = train_lbl,
            test_images = test_img,
            test_labels = test_lbl,
            fractions = fractions,
            num_runs= num_runs,
            **config # Unpacks rest of dictionary to kwargs
        )

if __name__ == "__main__":
    main()