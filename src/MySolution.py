import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

### TODO: import any other packages you need for your solution

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
        

    def plot_decision_boundary(self, X, y, title="Decision Boundary", size=22):
        h = 0.02 
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(title, fontsize=size)
        plt.xlabel("Feature 1", fontsize=size)
        plt.ylabel("Feature 2", fontsize=size)
        plt.show()
    
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of clusters
        self.labels = None
        self.cluster_centers_ = None  # Cluster centroids
        self.varience_threshold = 0.95

    def preprocess_data(self, X):
        """
        Apply PCA to reduce dimensionality.
        """
        # Fit PCA to the data
        self.pca = PCA()
        self.pca.fit(X)
        
        # Calculate cumulative variance
        cumulative_variance = self.pca.explained_variance_ratio_.cumsum()
        
        # Determine the number of components to cover the desired variance
        n_components = (cumulative_variance >= self.varience_threshold).argmax() + 1
        
        # Apply PCA with the determined number of components
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        print(f"Number of components selected: {n_components}")
        print(f"Explained variance covered: {cumulative_variance[n_components-1]:.2f}")
    
        return X_reduced

    def postprocess_centroids(self):
        """
        Transform centroids back to the original feature space.
        """
        if self.pca is not None:
            return self.pca.inverse_transform(self.cluster_centers_)
        return self.cluster_centers_

    def kmeans_plus_plus_init(self,X, K):
        """
        Initialize centroids using K-means++.

        Args:
            X (ndarray): Dataset of shape (N, M).
            K (int): Number of clusters.

        Returns:
            centroids (ndarray): Initialized centroids of shape (K, M).
        """
        N, M = X.shape
        centroids = []

        # Step 1: Randomly select the first centroid
        first_centroid_idx = np.random.choice(N)
        centroids.append(X[first_centroid_idx])

        # Step 2: Select remaining K-1 centroids
        for _ in range(1, K):
            # Compute distances from each point to the closest centroid
            distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])

            # Compute probabilities proportional to squared distances
            probabilities = distances / distances.sum()

            # Randomly select the next centroid based on probabilities
            next_centroid_idx = np.random.choice(N, p=probabilities)
            centroids.append(X[next_centroid_idx])

        return np.array(centroids)

    def formulate_lp(self, X, K):
        """
        Formulate the LP for soft clustering.
        """
        N, M = X.shape  # Number of samples and features
        n_vars = N * K  # Number of decision variables

        # Compute distance matrix
        distances = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                distances[i, j] = np.linalg.norm(X[i] - self.cluster_centers_[j])

        # Flatten the distance matrix for the objective function
        c = distances.flatten()
        c = c + 1e-6 # avoid inf, nan, div0

        # Equality constraint: Probabilities sum to 1 for each point
        A_eq = np.zeros((N, n_vars))
        for i in range(N):
            for j in range(K):
                A_eq[i, i * K + j] = 1
        b_eq = np.ones(N)

        # Bounds: 0 <= z_ij <= 1
        bounds = [(0, 1) for _ in range(n_vars)]

        return c, A_eq, b_eq, bounds

    
    def solve_lp(self, c, A_eq, b_eq, bounds):
        """
        Solve the LP using linprog.

        Args:
            c (ndarray): Coefficients for the objective function.
            A_eq (ndarray): Equality constraint matrix.
            b_eq (ndarray): Equality constraint vector.
            bounds (list): Bounds for decision variables.

        Returns:
            z (ndarray): Optimized decision variables.
        """
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not result.success:
            raise ValueError("LP solution failed.")
        return result.x
    
    def update_centroids(self, X, z, K):
        """
        Update centroids using soft assignments.
        """
        new_centroids = []
        z = z.reshape(X.shape[0], K)  # Reshape z to (N, K)
        for j in range(K):
            weights = z[:, j]
            weighted_sum = np.sum(weights[:, None] * X, axis=0)
            centroid = weighted_sum / weights.sum()
            new_centroids.append(centroid)
        return np.array(new_centroids)
    
    def train(self, trainX, max_iters=20, tol=1e-4):
        """
        Iterative LP-based clustering with PCA preprocessing.
        """
        # Apply PCA preprocessing
        trainX_reduced = self.preprocess_data(trainX)

        # Initialize centroids randomly in PCA-reduced space
        self.cluster_centers_ = trainX_reduced[np.random.choice(trainX_reduced.shape[0], self.K, replace=False)]

        for iteration in range(max_iters):
            # Formulate and solve LP for current centroids
            c, A_eq, b_eq, bounds = self.formulate_lp(trainX_reduced, self.K)
            z = self.solve_lp(c, A_eq, b_eq, bounds)
            z = z.reshape(trainX_reduced.shape[0], self.K)

            # Assign clusters
            self.labels = np.argmax(z, axis=1)

            # Update centroids
            new_centroids = self.update_centroids(trainX_reduced, z, self.K)

            # Check for convergence
            if np.linalg.norm(new_centroids - self.cluster_centers_) < tol:
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.cluster_centers_ = new_centroids
        
        return self.labels
    
    def infer_cluster(self, testX):
        """
        Assign new data points to the existing clusters and postprocess results.
        """
        # Transform test data using PCA
        testX_reduced = self.pca.transform(testX)

        # Assign clusters based on the nearest cluster centroid in reduced space
        pred_labels = np.array([
            np.argmin([np.linalg.norm(x - c) for c in self.cluster_centers_])
            for x in testX_reduced
        ])

        # Postprocess centroids back to original space for interpretation (optional)
        self.cluster_centers_original_ = self.postprocess_centroids()

        return pred_labels
        

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

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
        return data_to_label
    




##########################################################################
#--- Task 3 (Option 2) ---#
from gekko import GEKKO

class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        self.coords = np.zeros([num_features,2])  # x/y pixel coordinates for selected features


    def Covariance(self, X,y):
        N_features = X.shape[1]
        cov_vec = np.zeros(N_features)
        for i in range(X.shape[1]):
            x_i = X[:,i]
            xbar, ybar = x_i.mean(), y.mean()
            cov_vec[i] = np.sum((x_i-xbar)*(y-ybar))/(len(x_i)-1)
        return cov_vec


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        feat_to_keep = 0

        # If class labels array is NoneType, do clustering branch        
        if trainY is None:
            print("doing clustering feature selection")
            feat_to_keep = self.unsupervised_feature_selection(trainX, trainY)
            
        # Else, do classification branch
        else:
            print("doing classification feature selection")
            feat_to_keep = self.supervised_feature_selection(trainX, trainY)
            
        # Return an index list that specifies which features to keep
        return feat_to_keep


    def unsupervised_feature_selection(self, trainX, trainY): 
        # Compute pixel variances
        variance = np.var(trainX, axis=0)

        ## Formulate and run the Integer Linear Program
        mdl = GEKKO(remote=False)

        # Define max number of features
        K = mdl.Const(self.num_features)

        # Define covariance vector constants
        z = [mdl.Const(value=val) for val in variance]

        # Define t_min auxiliary variable
        t = mdl.Var(value=0, lb=0)

        # Define pixel selection vector, with its integrality and 0/1 constraints
        s = [mdl.Var(value=0, lb=0, ub=1, integer=True) for i in range(trainX.shape[1])]

        # Define constraints
        ## Max number of pixels constraint
        mdl.Equation(np.sum(s) <= K)
        ## Set t to the sum variance across all selected pixels
        mdl.Equation(t <= mdl.sum([s_i*z_i for (s_i, z_i) in zip(s,z)]))

        # Set objective function to maximize t
        mdl.Maximize(t)
        mdl.options.MAX_ITER = 10000
        mdl.options.SOLVER = 1    
        mdl.options.IMODE = 3
        # mdl.solve(disp=True)
        mdl.solve()

        # Get the 1/0 mask of selected features from the final values of s
        feature_mask = np.array([s_i.value for s_i in s])

        # Get feature indices of the selected features from the mask
        selected_feature_indices = np.nonzero(feature_mask)[0]
        return selected_feature_indices


    def supervised_feature_selection(self, trainX, trainY):
        # Get dataset unique labels and the sample indices corresponding to those labels
        unique_labels = np.unique(trainY)
        indices_per_label = []
        for i, label in enumerate(unique_labels):
            indices = np.where(trainY == label)
            indices_per_label.append(indices)

        # Assume no access to itertools, just use the known possible pairs given the fashion mnist dataset
        label_pairs = ((0,1), (0,2), (1,2))

        pairwise_covariances = []
        
        # Iterate across all potential label pairs
        for (lbl_idx_a, lbl_idx_b) in label_pairs:
            lbl_a = unique_labels[lbl_idx_a]
            lbl_b = unique_labels[lbl_idx_b]

            # Combine sample indices from both classes
            indices = np.append(indices_per_label[lbl_idx_a], indices_per_label[lbl_idx_b])

            # Get X and Y samples from the indices
            X_prime = np.take(trainX, indices, axis=0)
            Y_prime = np.take(trainY, indices)
            
            # print(X_prime.shape)
            # print(Y_prime.shape)

            # relabel original class labels to -1, +1
            for i in range(len(Y_prime)):
                if Y_prime[i] == lbl_a:
                    Y_prime[i] = 1
                elif Y_prime[i] == lbl_b:
                    Y_prime[i] = -1
            
            # compute covariance for the pair of labels
            pair_covariance = self.Covariance(X_prime, Y_prime)
            pairwise_covariances.append(np.absolute(pair_covariance))
        
        # Formulate and run the integer linear program
        mdl = GEKKO(remote=False)

        ### Define constants and variables
        # Define max features
        K = mdl.Const(self.num_features)
        # Define covariance vector constants
        z_0_1 = [mdl.Const(value=val) for val in pairwise_covariances[0]]
        z_0_2 = [mdl.Const(value=val) for val in pairwise_covariances[1]]
        z_1_2 = [mdl.Const(value=val) for val in pairwise_covariances[2]]
        # Define t_min auxiliary variable
        t_min = mdl.Var(value=0, lb=0)
        # Define pixel selection vector, with its integrality and 0/1 constraints
        s = [mdl.Var(value=0, lb=0, ub=1, integer=True) for i in range(trainX.shape[1])]
        # s = [mdl.Var(value=0, lb=0, ub=1) for i in range(trainX.shape[1])]
        # print(len(s))

        ### Define constraints
        ## Max number of pixels constraint
        mdl.Equation(np.sum(s) <= K)
        ## Set t_min to the min sum absolute covariance across all class pairs
        # mdl.Equation(t_min <= np.dot(s, z_0_1))
        # mdl.Equation(t_min <= np.dot(s, z_0_2))
        # mdl.Equation(t_min <= np.dot(s, z_1_2))
        ## Reformulate dot products as element-wise multiply and sum to avoid errors in GEKKO
        mdl.Equation(t_min <= mdl.sum([s_i*z_0_1_i for (s_i, z_0_1_i) in zip(s,z_0_1)]))
        mdl.Equation(t_min <= mdl.sum([s_i*z_0_2_i for (s_i, z_0_2_i) in zip(s,z_0_2)]))
        mdl.Equation(t_min <= mdl.sum([s_i*z_1_2_i for (s_i, z_1_2_i) in zip(s,z_1_2)]))

        # Set objective function to maximize t_min
        mdl.Maximize(t_min)
        mdl.options.MAX_ITER = 10000
        mdl.options.SOLVER = 1    
        mdl.options.IMODE = 3
        # mdl.solve(disp=True)
        mdl.solve()

        # Get the 1/0 mask of selected features from the final values of s
        feature_mask = np.array([s_i.value for s_i in s])
        
        # Get feature indices of the selected features from the mask
        selected_feature_indices = np.nonzero(feature_mask)[0]
        
        return selected_feature_indices

    
    
    
