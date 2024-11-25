import numpy as np
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional

#--- Task 1 ---#
class MyClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, 
                max_iter: int = 100,
                tol: float = 1e-8,
                lambda_reg: float = 0.1):  # Added regularization parameter
        
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_reg = lambda_reg

        self.weights = []
        self.biases = []
        self.scaler = StandardScaler()
        self.classes_ = None
        self.n_classes_ = None
        
        solvers.options['maxiters'] = max_iter
        solvers.options['abstol'] = tol
        solvers.options['reltol'] = tol
        solvers.options['show_progress'] = False


    def train(self, trainX, trainY):
        def _solve_binary_problem(trainX, trainY) -> Tuple[np.ndarray, float]:
            n_samples, n_features = trainX.shape
            
            # For L1 regularization, we need two variables (u,v) for each weight
            # Total variables: n_features (w) + 1 (b) + n_samples (xi) + 2*n_features (u,v for L1)
            n_vars = n_features + 1 + n_samples + 2*n_features
            
            # Objective: minimize λ(u + v) + Σξ
            c = matrix(np.hstack([
                np.zeros(n_features + 1),          # coefficients for w and b
                np.ones(n_samples),                # coefficients for slack variables
                self.lambda_reg * np.ones(2*n_features)  # coefficients for u,v (L1 terms)
            ]))

            # First set of constraints: -y_i(w^T x_i + b) + ξ_i ≤ -1
            G1 = np.zeros((n_samples, n_vars))
            G1[:, :n_features] = -trainY.reshape(-1, 1) * trainX
            G1[:, n_features] = -trainY
            np.fill_diagonal(G1[:, n_features + 1:n_features + 1 + n_samples], -1)
            
            # Second set of constraints: w = u - v
            G2 = np.zeros((n_features, n_vars))
            G2[:, :n_features] = np.eye(n_features)  # coefficients for w
            G2[:, n_features + 1 + n_samples:n_features + 1 + n_samples + n_features] = -np.eye(n_features)  # for u
            G2[:, n_features + 1 + n_samples + n_features:] = np.eye(n_features)  # for v
            
            # Third set of constraints: -w = -u + v
            G3 = -G2
            
            # Fourth set of constraints: non-negativity for slack and L1 variables
            G4 = np.zeros((n_samples + 2*n_features, n_vars)) + 1e-8
            np.fill_diagonal(G4[:n_samples, n_features + 1:n_features + 1 + n_samples], -1)  # for xi
            np.fill_diagonal(G4[n_samples:, n_features + 1 + n_samples:], -1)  # for u,v
            
            # Combine all constraints
            G = matrix(np.vstack([G1, G2, G3, G4]))
            
            # Right hand side of constraints
            h = matrix(np.hstack([
                -np.ones(n_samples),                # for first constraint
                np.zeros(2*n_features),             # for w = u - v constraints
                1e-8 * np.ones(n_samples + 2*n_features)  # for non-negativity
            ]))

            # Solve the LP problem
            solution = solvers.lp(c, G, h)
            
            # Extract solution
            sol_vector = np.array(solution['x']).flatten()
            return sol_vector[:n_features], sol_vector[n_features]
            
        # Scale features
        X_scaled = self.scaler.fit_transform(trainX)
        
        # Get unique classes
        self.classes_ = np.unique(trainY)
        self.n_classes_ = len(self.classes_)
        
        # Reset weights and biases
        self.weights = []
        self.biases = []
        
        # Train binary classifiers
        for class_label in self.classes_:
            y_binary = np.where(trainY == class_label, 1, -1)
            w, b = _solve_binary_problem(X_scaled, y_binary)
            self.weights.append(w)
            self.biases.append(b)
        
        # Convert to numpy arrays
        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)
        
        # Compute feature importances
        self.feature_importances_ = np.abs(self.weights).mean(axis=0)
        self.feature_importances_ /= self.feature_importances_.sum()
        
        return self
    
    def predict(self, testX):
        X_scaled = self.scaler.transform(testX)
        scores = X_scaled @ self.weights.T + self.biases
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            
        return self.classes_[probs.argmax(axis=1)]

    
    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    
########################### EXTRA FUNCTIONS FOR TASK 1  ###########################

    def plot_confusion_matrix(self, predY, testY):
        cm = confusion_matrix(testY, predY)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        

    def plot_decision_boundary(self, X, y, title="Decision Boundary"):
        h = 0.02 
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    

##########################################################################
#--- Task 2 ---#
##########################################################################
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''


        # Update and teturn the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # Return the cluster labels of the input data (testX)
        return None #pred_labels

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 (Option 1) ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return None #data_to_label
    




##########################################################################
#--- Task 3 (Option 2) ---#
class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        


        # Return an index list that specifies which features to keep
        return None #feat_to_keep
    
    