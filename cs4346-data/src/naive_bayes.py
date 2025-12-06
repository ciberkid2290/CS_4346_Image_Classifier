# Goal: Implement a Naive Bayes classifier for digit recognition
import numpy as np

class NaiveBayesClassifier:
    """
    Bernoulli Naive Bayes classifier for digit recognition.
    Calculates the probabilities baed on the presence or absence of features.
    It uses log probabilities to avoid numerical underflow issues and improve stability.
    """

    def __init__(self, num_classes, smoothing = 1.0):
        """
        Initializes the model.
        Args:
            num_classes (int): Number of possible class labels (10 for digits, 2 for faces).
            smoothing (float): Laplace smoothing parameter.
        """
        self.num_classes = num_classes
        self.smoothing = smoothing

        # Model parameters
        self.num_features = None
        self.class_log_prior = None  # Log prior probabilities for each class
        self.feature_log_prob = None  # Log probabilities of features given class
        self.feature_log_neg_prob = None  # Log probabilities of feature absence given class

    def fit(self, X, y):

        """
        Initializes the Naive bayes parameters from the training data.
        Args:
            x (np.ndarray): Training data of shape (num_samples, num_features), where each feature is binary (0 or 1).
            y (np.ndarray): Class labels of shape (num_samples,), with integer class labels from 0 to num_class.
        """
        N, D = X.shape # N = number of samples, D = number of features
        self.num_features = D

        # Count occurrences of each class
        class_counts = np.zeros(self.num_classes, dtype=np.float64)
        feature_counts = np.zeros((self.num_classes, D), dtype=np.float64) # feature_counts[c, j] = count of feature j in class c

        for i in range(N):
            label = y[i]
            class_counts[label] += 1
            feature_counts[label] += X[i] 

        # Calculate log prior probabilities: log P(class)
        # P(c) = (count(c) + smoothing) / (N + smoothing * num_classes)
        sm = self.smoothing
        self.class_log_prior = np.log((class_counts + sm) / (N + sm * self.num_classes))

        # Calculate log conditional probabilities: log P(feature | class)
        denominator = class_counts[:, np.newaxis] + 2.0 * sm # has 2 * sm because feature is binary (0 or 1)

        # P(feature_j = 1 | class = c)
        prob_one = (feature_counts + sm) / denominator
        self.feature_log_prob = np.log(prob_one)

        # P(feature_j = 0 | class = c) = 1 - P(feature_j = 1 | class = c)
        self.feature_log_neg_prob = np.log(1.0 - prob_one)

    def _joint_log_likelihood(self, X):
        """
        Computes the joint log-likelihood for each sample.
        This is proportional to log P(class) + sum(log P(feature | class))
        
        Args:
            X (np.ndarray): Data of shape (N, D) to compute log-likelihoods for.
        
        Returns:
            np.ndarray: Array of shape (N, num_classes) where entry (i, c) is the joint log-likelihood of sample i for class c.
        """
        # Term for features that are 1: X * log P(X=1|C)
        term_for_ones = X @ self.feature_log_prob.T

        # Term for features that are 0: (1 - X) * log P(X=0|C)
        term_for_zeros = (1 - X) @ self.feature_log_neg_prob.T

        # The total log-likelihood
        joint_log_likelihood = self.class_log_prior + term_for_ones + term_for_zeros
        return joint_log_likelihood
    
    def predict(self, X):
        """
        Predicts class labels for a set of feature vectors.
        
        Args:
            X (np.ndarray): Data of shape (N, D) to predict.
            
        Returns:
            np.ndarray: A 1D array of shape (N,) containing the predicted class
            labels for each sample in X.
        """
        # Ensure model is trained
        if self.class_log_prior is None:
            raise RuntimeError("You must call fit() before predict().")
        
        # Calculate the log-likelihood of each class for each sample.
        log_likelihoods = self._joint_log_likelihood(X)

        # The predicted class is the one with the highest log-likelihood.
        return np.argmax(log_likelihoods, axis=1)

    def predict_one(self, X):
        """
        Convenience method to predict the class label for a single feature vector.
        
        Args:
            X (np.ndarray): A 1D array of shape (D,) representing a single feature vector.
        
        Returns:
            int : The predicted class label for the input feature vector.
        """
        X = np.asarray(X).reshape(1, -1)  # Reshape to (1, D)
        return int(self.predict(X)[0])