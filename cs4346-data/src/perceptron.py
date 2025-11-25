# Implement a Perceptron classifier for digit recognition
import numpy as np

class PerceptronClassifier:
    # Initialize classifier
    def __init__(self, num_classes, num_features, learning_rate = 1.0):
        self.learning_rate = learning_rate # Learning rate for weight updates
        self.num_classes = num_classes # Number of distinct classes
        # Initialize weights to zeros
        self.weights = np.zeros((num_classes, num_features))

    # Train the model with provided data
    def train(self, features, labels, epochs = 10):
        for epoch in range(epochs):
            for feature_vector, true_label in zip(features, labels): 
                # Calculate scores for each class
                # Scores = w * x
                scores = self.weights.dot(feature_vector)

                # Make prediction
                predicted_label = np.argmax(scores)

                # Update weights if prediction is incorrect
                if predicted_label != true_label:
                    # Strengthen connection with true class, then weaken connection with predicted class
                    self.weights[true_label, :] += self.learning_rate * feature_vector
                    self.weights[predicted_label, :] -= self.learning_rate * feature_vector
    
    # Predict the class for a set of feature vectors
    def predict(self, features):
        pass