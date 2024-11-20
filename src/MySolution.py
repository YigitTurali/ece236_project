import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from pyomo.environ import *
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
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # Initialize parameters
        self.cluster_centers_ = None
        self.pca = None
        self.V = None  # PCA projection matrix
        self.mu = None  # Data mean for centering
        self.tilde_M = None  # Reduced dimension after PCA
        self.alpha = 0.95  # PCA variance ratio
        
    def pca_preprocessing(self, X):
        """PCA preprocessing step"""
        # Center the data
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu
        
        # Perform PCA
        self.pca = PCA(n_components=self.alpha)
        X_tilde = self.pca.fit_transform(X_centered)
        self.V = self.pca.components_.T
        self.tilde_M = X_tilde.shape[1]
        
        return X_tilde
    
    def solve_lp_clustering(self, X_tilde):
        """
        Solve LP clustering problem using Pyomo.
        Args:
            X_tilde (ndarray): Data points (N x M).
            K (int): Number of clusters.

        Returns:
            Z (ndarray): Cluster assignments (soft, N x K).
            C (ndarray): Centroids (K x M).
        """
        N, M = X_tilde.shape  # N: number of data points, M: data dimension

        # Create a Pyomo model
        model = ConcreteModel()

        # Decision variables
        model.Z = Var(range(N), range(self.K), bounds=(0, 1))  # Soft cluster assignments
        model.C = Var(range(self.K), range(M), domain=Reals)  # Centroids
        model.U = Var(range(N), range(self.K), range(M), bounds=(0, None))  # Positive part
        model.V = Var(range(N), range(self.K), range(M), bounds=(0, None))  # Negative part
        model.D = Var(range(N), range(self.K), bounds=(0, None))  # Distances

        # Objective function: Minimize weighted distances
        model.obj = Objective(
            expr=sum(model.Z[i, j] * model.D[i, j] for i in range(N) for j in range(self.K)),
            sense=minimize
        )

        # Constraints

        # Each point is assigned to exactly one cluster
        model.assignment = ConstraintList()
        for i in range(N):
            model.assignment.add(sum(model.Z[i, j] for j in range(self.K)) == 1)

        # Each cluster must have at least one point
        model.cluster = ConstraintList()
        for j in range(self.K):
            model.cluster.add(sum(model.Z[i, j] for i in range(N)) >= 1)

        # Distance constraints
        model.distance_constraints = ConstraintList()
        for i in range(N):
            for j in range(self.K):
                # Distance calculation: D[i, j] = sum(U[i, j, :] + V[i, j, :])
                model.distance_constraints.add(
                    model.D[i, j] == sum(model.U[i, j, k] + model.V[i, j, k] for k in range(M))
                )

                for k in range(M):
                    # X_tilde[i, k] - C[j, k] = U[i, j, k] - V[i, j, k]
                    model.distance_constraints.add(
                        X_tilde[i, k] - model.C[j, k] == model.U[i, j, k] - model.V[i, j, k]
                    )

        # Solver
        solver = SolverFactory('glpk')  # Default open-source solver
        result = solver.solve(model, tee=True)

        if result.solver.termination_condition != TerminationCondition.optimal:
            raise ValueError(f"Optimization failed: {result.solver.termination_condition}")

        # Extract results
        Z = np.array([[model.Z[i, j].value for j in range(self.K)] for i in range(N)])
        C = np.array([[model.C[j, k].value for k in range(M)] for j in range(self.K)])

        return Z, C
    
    def post_processing(self, Z, C, X_tilde):
        """Post-processing to get hard assignments"""
        N = X_tilde.shape[0]
        labels = np.argmax(Z, axis=1)
        
        # Update centroids
        for j in range(self.K):
            mask = labels == j
            if np.sum(mask) > 0:
                C[j] = np.mean(X_tilde[mask], axis=0)
            
            return labels, C
    
    def train(self, trainX):
        """Main training function"""
        # Step 1: PCA preprocessing
        X_tilde = self.pca_preprocessing(trainX)
        print(f"Reduced dimension: {X_tilde}")
        
        # Step 2: Solve LP
        Z, C = self.solve_lp_clustering(X_tilde)
        print(f"Cluster centers: {C}")
        print(f"Cluster assignments: {Z}")
        
        # Step 3: Post-processing
        self.labels, self.cluster_centers_ = self.post_processing(Z, C, X_tilde)
        print(f"Final cluster centers: {self.cluster_centers_}")
        print(f"Final cluster assignments: {self.labels}")
        return self.labels
    
    
    def infer_cluster(self, testX):
        """Assign new points to clusters"""
        # Project test data
        X_centered = testX - self.mu
        X_tilde = self.pca.transform(X_centered)
        
        # Compute distances to centroids
        N = X_tilde.shape[0]
        distances = np.zeros((N, self.K))
        
        for i in range(N):
            for j in range(self.K):
                # Compute L1 distance
                distances[i, j] = np.sum(np.abs(X_tilde[i] - self.cluster_centers_[j]))
        
        # Assign to nearest centroid
        pred_labels = np.argmin(distances, axis=1)
        
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
    
    