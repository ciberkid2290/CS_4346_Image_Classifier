# Goal: Implement a Naive Bayes classifier for digit recognition
import numpy as np

class NaiveBayesClassifier:
    # Initialize the classifier with smoothing parameter k
    def __init__(self, smoothing_k = 1):
        self.k = smoothing_k # Laplace smoothing parameter
        self.priors = {} # Prior probabilities for each class
        self.conditional_probs = {} # Conditional probabilities P(feature|class)
        self.classes = [] # List of unique classes
    
    # Train the model with the given features and labels
    def train(self, features, labels):
        num_samples, num_features = features.shape
        self.classes = np.unique(labels)

        # Calculate prior probabilities: P(class)
        for c in self.classes:
            self.priors[c] = (np.sum(labels == c) + self.k) / (num_samples + self.k * len(self.classes))
        

        # Calculate conditional probabilities
        for c in self.classes:
            # Get all features for current class
            class_features = features[labels == c]
            # Sum of features column-wise (occurrences of feature being 1)
            feature_counts = np.sum(class_features, axis=0)
            total_features_in_class = len(class_features)

            # Apply Laplace smoothing
            self.conditional_probs[c] = (feature_counts + self.k) / (total_features_in_class + 2 * self.k)
    
    # Predict the class for a set of feature vectors
    def predict(self, features):
        pass