# Implement a Perceptron classifier for digit recognition
import numpy as np

class PerceptronClassifier:
    """
    Multi-class Perceptron classifier.
    Uses the Perceptron learning algorithm to classify input feature vectors
    into one of several classes based on learned weights.
    """
    # Initialize classifier
    def __init__(self, num_classes, num_features = None, learning_rate = 1.0, max_epochs = 10, shuffle = True):
        """
        Initialize the Perceptron model.
        Args:
            num_classes (int): Number of possible class labels (10 for digits, 2 for faces).
            num_features (int or None): Number of features in the input data.
            learning_rate (float): Step size for updates.
            max_epochs (int): Number of passes (epochs) over the training data.
            shuffle (bool): Whether to shuffle training data each epoch.
        """
        self.num_classes = num_classes # Number of distinct classes
        self.num_features = num_features
        self.learning_rate = learning_rate # Learning rate for weight updates
        self.max_epochs = max_epochs
        self.shuffle = shuffle
        
        # Weight matrix and bias vector to be initialized in fit() when num_features is known
        self.weights = None
        self.bias = None
    
    # --------------------------------
    # Training
    # --------------------------------
    def fit(self, X, y):
        """
        Train the Perceptron model using the provided training data.
        
        Args:
            X (np.ndarray): Training data of shape (num_samples, num_features).
            y (np.ndarray): Class labels of shape (num_samples,), with integer labels from 0 to num_classes - 1.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        N, D = X.shape # N = number of samples, D = number of features

        # Infer num_features from data if not provided
        if self.num_features is None:
            self.num_features = D
        
        # Initialize weights and bias
        if self.weights is None:
            self.weights = np.zeros((self.num_classes, self.num_features), dtype=np.float64)
        if self.bias is None:
            self.bias = np.zeros(self.num_classes, dtype=np.float64)
        
        # Training loop
        for epoch in range(self.max_epochs):
            indices = np.arange(N)
            if self.shuffle:
                np.random.shuffle(indices)
            
            for i in indices:
                x_i = X[i]
                y_true = y[i]

                # Compute scores: scores_c = w_c . x_i + b_c
                scores = self.weights.dot(x_i) + self.bias
                y_pred = np.argmax(scores) # Predicted class

                # If prediction is incorrect, update weights and biases
                if y_pred != y_true:
                    lr = self.learning_rate
                    # Update weights
                    self.weights[y_true] += lr * x_i
                    self.weights[y_pred] -= lr * x_i
                    # Update biases
                    self.bias[y_true] += lr
                    self.bias[y_pred] -= lr
        
    
    
    # Predict the class for a set of feature vectors
    def predict(self, X):
        # Calculate scores for each class for all feature vectors
        scores = X.dot(self.weights.T) + self.bias

        # For each example, pick the class with the highest score
        predicted_labels = np.argmax(scores, axis=1)
        return predicted_labels
    
    # Predict the class for a single feature vector
    def predict_one(self, x):
        X = np.asarray(x).reshape(1, -1) # Reshape to 2D array (1, D)
        return int(self.predict(X)[0])