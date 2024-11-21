import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import linprog
from sklearn.decomposition import PCA

### TODO: import any other packages you need for your solution

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.w = None
        # self.b = None

    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''


        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of clusters
        self.labels = None
        self.cluster_centers_ = None  # Cluster centroids
        self.varience_threshold = 0.98

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

        # Equality constraint: Probabilities sum to 1 for each point
        A_eq = np.zeros((N, n_vars))
        for i in range(N):
            for j in range(K):
                A_eq[i, i * K + j] = 1
        b_eq = np.ones(N)

        # Bounds: 0 <= z_ij <= 1
        bounds = [(0, 1) for _ in range(n_vars)]

        # Add entropy terms
        gamma = 0.2
        cluster_sizes = np.sum(self.z_prev, axis=0) if hasattr(self, 'z_prev') else np.ones(K)/K
        entropy_penalty = -gamma * np.log(cluster_sizes + 1e-10)
        c = c.reshape(N, K) + entropy_penalty
        c = c.flatten()

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
    
    def train_reduced(self, trainX, max_iters=20, tol=1e-4):
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

    
    def train(self, trainX, max_iters=20, tol=1e-4):
        """
        Task 2-2: Iterative LP-based clustering.
        
        Args:
            trainX (ndarray): Training data of shape (N, M).
            max_iters (int): Maximum number of iterations for centroid refinement.
            tol (float): Convergence tolerance for centroids.
        """

        # Initialize centroids randomly
        self.cluster_centers_ = self.kmeans_plus_plus_init(trainX, self.K)

        for iteration in range(max_iters):
            # Step 1: Formulate and solve LP for current centroids
            c, A_eq, b_eq, bounds = self.formulate_lp(trainX, self.K)
            z = self.solve_lp(c, A_eq, b_eq, bounds)
            z = z.reshape(trainX.shape[0], self.K)

            # Step 2: Assign clusters
            self.labels = np.argmax(z, axis=1)

            # Step 3: Update centroids
            new_centroids = []
            for cluster_id in range(self.K):
                cluster_points = trainX[self.labels == cluster_id]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    # Handle empty cluster
                    new_centroids.append(self.cluster_centers_[cluster_id])  # Keep old centroid

            new_centroids = np.array(new_centroids)

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
class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        


        # Return an index list that specifies which features to keep
        return feat_to_keep
    
    