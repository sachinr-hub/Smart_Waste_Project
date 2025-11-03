import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib
import os
import base64
from io import BytesIO

class WasteNeighborhoodClustering:
    """KNN-based clustering model for identifying similar neighborhood waste profiles."""
    
    def __init__(self, n_clusters=4, n_neighbors=5):
        """
        Initialize the clustering model.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to use in KMeans
        n_neighbors : int
            Number of neighbors to use in KNN
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained = False
        self.silhouette_avg = None
        self.cluster_centers = None
        self.cluster_sizes = None
        self.X_scaled = None
        
    def fit(self, X, find_optimal_clusters=False, max_clusters=10):
        """
        Fit the clustering model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        find_optimal_clusters : bool
            Whether to find the optimal number of clusters using silhouette score
        max_clusters : int
            Maximum number of clusters to consider when find_optimal_clusters is True
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            neighborhood_ids = X['neighborhood_id'].values if 'neighborhood_id' in X.columns else None
            # Remove neighborhood_id if present - it's an identifier, not a feature
            if 'neighborhood_id' in X.columns:
                X = X.drop('neighborhood_id', axis=1)
                # Put it back in feature_names for later reference
                self.feature_names.remove('neighborhood_id')
            X_values = X.values
        else:
            X_values = X
            neighborhood_ids = None
            
        # Scale the data
        self.X_scaled = self.scaler.fit_transform(X_values)
        
        # Find optimal number of clusters if requested
        if find_optimal_clusters:
            print("Finding optimal number of clusters...")
            best_n_clusters = self._find_optimal_clusters(self.X_scaled, max_clusters)
            print(f"Optimal number of clusters: {best_n_clusters}")
            self.n_clusters = best_n_clusters
            self.kmeans_model = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        
        # Fit KMeans model
        print(f"Clustering neighborhoods into {self.n_clusters} groups...")
        self.kmeans_model.fit(self.X_scaled)
        
        # Get cluster assignments and calculate silhouette score
        cluster_labels = self.kmeans_model.labels_
        if len(np.unique(cluster_labels)) > 1:  # Silhouette score requires at least 2 clusters
            self.silhouette_avg = silhouette_score(self.X_scaled, cluster_labels)
            print(f"Silhouette Score: {self.silhouette_avg:.4f}")
        
        # Store cluster centers and sizes
        self.cluster_centers = self.kmeans_model.cluster_centers_
        self.cluster_sizes = np.bincount(cluster_labels)
        
        # Create a new dataset with cluster assignments
        if neighborhood_ids is not None:
            self.neighborhood_clusters = pd.DataFrame({
                'neighborhood_id': neighborhood_ids,
                'cluster': cluster_labels
            })
        else:
            self.neighborhood_clusters = pd.DataFrame({
                'sample_id': np.arange(len(cluster_labels)),
                'cluster': cluster_labels
            })
            
        # Train KNN model on cluster assignments for future predictions
        self.knn_model.fit(self.X_scaled, cluster_labels)
        
        # Print cluster sizes
        for i, size in enumerate(self.cluster_sizes):
            print(f"Cluster {i}: {size} neighborhoods ({size / len(cluster_labels) * 100:.1f}%)")
        
        self.trained = True
        return self
    
    def _find_optimal_clusters(self, X, max_clusters):
        """
        Find the optimal number of clusters using silhouette score.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
        max_clusters : int
            Maximum number of clusters to consider
            
        Returns:
        --------
        best_n_clusters : int
            Optimal number of clusters
        """
        silhouette_scores = []
        
        # Try different numbers of clusters
        for n_clusters in range(2, min(max_clusters + 1, len(X) // 2)):
            # Initialize and fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            print(f"With {n_clusters} clusters, silhouette score: {silhouette_avg:.4f}")
        
        # Select the number of clusters with the highest silhouette score
        best_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
        
        return best_n_clusters
    
    def predict_cluster(self, X):
        """
        Predict cluster assignment for new data points.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New samples
            
        Returns:
        --------
        cluster_labels : array-like of shape (n_samples,)
            Predicted cluster labels
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            # Remove neighborhood_id if present
            if 'neighborhood_id' in X.columns:
                X = X.drop('neighborhood_id', axis=1)
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Use KNN to predict cluster
        return self.knn_model.predict(X_scaled)
    
    def find_similar_neighborhoods(self, neighborhood_id=None, X=None, n_neighbors=None):
        """
        Find neighborhoods similar to the given one using KNN.
        
        Parameters:
        -----------
        neighborhood_id : int or None
            ID of the neighborhood to find similar ones for.
            If provided, will look up this ID in the training data.
        X : array-like of shape (1, n_features) or None
            Feature vector of a neighborhood. Required if neighborhood_id is None.
        n_neighbors : int or None
            Number of similar neighborhoods to return. If None, uses self.n_neighbors.
            
        Returns:
        --------
        similar_neighborhoods : list
            IDs of similar neighborhoods
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        if neighborhood_id is not None:
            # Find the neighborhood in the training data
            if not hasattr(self, 'neighborhood_clusters'):
                raise ValueError("Model was not trained with neighborhood IDs")
                
            if neighborhood_id not in self.neighborhood_clusters['neighborhood_id'].values:
                raise ValueError(f"Neighborhood ID {neighborhood_id} not found in training data")
                
            # Get the index of this neighborhood in the original data
            idx = np.where(self.neighborhood_clusters['neighborhood_id'].values == neighborhood_id)[0][0]
            
            # Get its feature vector
            X_query = self.X_scaled[idx].reshape(1, -1)
        elif X is not None:
            # Scale the input data
            if isinstance(X, pd.DataFrame):
                # Remove neighborhood_id if present
                if 'neighborhood_id' in X.columns:
                    X = X.drop('neighborhood_id', axis=1)
                X = X.values
                
            X_query = self.scaler.transform(X)
        else:
            raise ValueError("Either neighborhood_id or X must be provided")
            
        # Use KNN to find similar neighborhoods
        distances, indices = self.knn_model.kneighbors(X_query, n_neighbors=n_neighbors+1)
        
        # First index is the query point itself if it was in the training data, so skip it
        if neighborhood_id is not None:
            indices = indices[0][1:]
        else:
            indices = indices[0]
            
        # Get the neighborhood IDs
        if hasattr(self, 'neighborhood_clusters'):
            similar_neighborhoods = self.neighborhood_clusters['neighborhood_id'].values[indices]
        else:
            similar_neighborhoods = indices
            
        return similar_neighborhoods.tolist()
    
    def get_cluster_profiles(self):
        """
        Generate profiles of each cluster based on feature means.
        
        Returns:
        --------
        cluster_profiles : DataFrame
            Mean values of features for each cluster
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        # Convert cluster centers back to original scale
        centers_original = self.scaler.inverse_transform(self.cluster_centers)
        
        # Create a DataFrame with cluster profiles
        if self.feature_names:
            cluster_profiles = pd.DataFrame(
                centers_original,
                columns=self.feature_names
            )
        else:
            cluster_profiles = pd.DataFrame(
                centers_original,
                columns=[f"feature_{i}" for i in range(centers_original.shape[1])]
            )
            
        # Add cluster info
        cluster_profiles['cluster'] = np.arange(self.n_clusters)
        cluster_profiles['size'] = self.cluster_sizes
        cluster_profiles['percentage'] = self.cluster_sizes / np.sum(self.cluster_sizes) * 100
        
        return cluster_profiles
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        model_data = {
            'kmeans_model': self.kmeans_model,
            'knn_model': self.knn_model,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'n_neighbors': self.n_neighbors,
            'feature_names': self.feature_names,
            'silhouette_avg': self.silhouette_avg,
            'cluster_centers': self.cluster_centers,
            'cluster_sizes': self.cluster_sizes,
            'trained': self.trained
        }
        
        if hasattr(self, 'neighborhood_clusters'):
            model_data['neighborhood_clusters'] = self.neighborhood_clusters
            
        if hasattr(self, 'X_scaled'):
            model_data['X_scaled'] = self.X_scaled
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            n_clusters=model_data['n_clusters'],
            n_neighbors=model_data['n_neighbors']
        )
        instance.kmeans_model = model_data['kmeans_model']
        instance.knn_model = model_data['knn_model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.silhouette_avg = model_data['silhouette_avg']
        instance.cluster_centers = model_data['cluster_centers']
        instance.cluster_sizes = model_data['cluster_sizes']
        instance.trained = model_data['trained']
        
        if 'neighborhood_clusters' in model_data:
            instance.neighborhood_clusters = model_data['neighborhood_clusters']
            
        if 'X_scaled' in model_data:
            instance.X_scaled = model_data['X_scaled']
        
        return instance
    
    def visualize_clusters(self, method='pca'):
        """
        Visualize the clusters in 2D using dimensionality reduction.
        
        Parameters:
        -----------
        method : str
            Method to use for dimensionality reduction ('pca' or 'tsne')
            
        Returns:
        --------
        plt_base64 : str
            Base64 encoded PNG image of the plot
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        # Reduce dimensionality for visualization
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(self.X_scaled)
            method_name = 'PCA'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.X_scaled)-1))
            X_2d = reducer.fit_transform(self.X_scaled)
            method_name = 't-SNE'
        else:
            raise ValueError("Method must be either 'pca' or 'tsne'")
            
        # Get cluster assignments
        cluster_labels = self.kmeans_model.labels_
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot clusters
        plt.subplot(2, 1, 1)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, 
                             cmap='viridis', alpha=0.7, s=50)
        
        # Plot cluster centers if using PCA
        if method == 'pca':
            centers_2d = reducer.transform(self.cluster_centers)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, alpha=0.8, marker='X')
            
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Neighborhood Clusters Visualization ({method_name})')
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        plt.grid(alpha=0.3)
        
        # Create a cluster profile summary
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        cluster_profiles = self.get_cluster_profiles()
        profile_text = "Cluster Profiles Summary:\n\n"
        
        for i in range(self.n_clusters):
            profile = cluster_profiles[cluster_profiles['cluster'] == i]
            profile_text += f"Cluster {i} ({profile['size'].values[0]} neighborhoods, {profile['percentage'].values[0]:.1f}%):\n"
            
            # Select top features for this cluster (highest and lowest relative to other clusters)
            if self.feature_names:
                feature_values = profile[self.feature_names].values[0]
                feature_names = self.feature_names
                
                # Normalize feature values across clusters for comparison
                all_features = cluster_profiles[self.feature_names].values
                feature_means = np.mean(all_features, axis=0)
                feature_stds = np.std(all_features, axis=0)
                
                # Calculate z-scores
                z_scores = (feature_values - feature_means) / (feature_stds + 1e-10)
                
                # Get top 3 highest and lowest features
                top_high_indices = np.argsort(z_scores)[-3:][::-1]
                top_low_indices = np.argsort(z_scores)[:3]
                
                # Add to profile text
                profile_text += "  Distinctive high features: "
                for idx in top_high_indices:
                    profile_text += f"{feature_names[idx]} ({z_scores[idx]:.2f}σ), "
                profile_text = profile_text[:-2] + "\n"
                
                profile_text += "  Distinctive low features: "
                for idx in top_low_indices:
                    profile_text += f"{feature_names[idx]} ({z_scores[idx]:.2f}σ), "
                profile_text = profile_text[:-2] + "\n\n"
                
        plt.text(0.05, 0.95, profile_text, fontsize=11, verticalalignment='top', 
                horizontalalignment='left', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        
        # Convert PNG to base64 string
        plt_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plt.close()
        
        return plt_base64


# Example usage
if __name__ == "__main__":
    # Load the data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data', 'knn_waste_dataset.csv')
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Create and train the model
        model = WasteNeighborhoodClustering(n_clusters=4, n_neighbors=5)
        model.fit(data, find_optimal_clusters=True, max_clusters=8)
        
        # Save the model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'kmeans_clustering_model.joblib')
        model.save_model(model_path)
        
        # Visualize clusters using PCA
        viz_pca = model.visualize_clusters(method='pca')
        
        # Save visualization
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'webapp', 'static', 'images')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        with open(os.path.join(viz_dir, 'kmeans_clusters_pca.png'), 'wb') as f:
            f.write(base64.b64decode(viz_pca))
        
        # Visualize clusters using t-SNE
        viz_tsne = model.visualize_clusters(method='tsne')
        
        with open(os.path.join(viz_dir, 'kmeans_clusters_tsne.png'), 'wb') as f:
            f.write(base64.b64decode(viz_tsne))
        
        # Generate and save cluster profiles
        profiles = model.get_cluster_profiles()
        profiles_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'data', 'cluster_profiles.csv')
        profiles.to_csv(profiles_path, index=False)
        
        print("KNN clustering model training and visualization complete!")
    else:
        print(f"Data file not found: {data_path}")
        print("Please generate the dataset first by running the generate_dataset.py script.") 