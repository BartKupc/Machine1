import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import json

# Add base directory to path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.feature_calculator import calculate_all_features
from utils.bitget_futures import BitgetFutures

class MarketClustering:
    def __init__(self, config=None):
        """Initialize the market clustering model"""
        self.base_dir = base_dir
        self.config = config
        self.pca_data = None
        self.kmeans_model = None
        self.cluster_stats = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Ensure analysis directory exists
        (self.base_dir / 'analysis').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'models').mkdir(parents=True, exist_ok=True)
        
        # If config is provided, initialize Bitget client
        if config and 'bitget_client' in config:
            self.bitget = config['bitget_client']
        
    def load_pca_model(self, model_path=None):
        """Load the saved PCA model"""
        try:
            # If no specific path provided, find the latest
            if model_path is None:
                model_path = self.find_latest_pca_model()
                
            # If we still don't have a path, exit
            if model_path is None:
                return False
                
            logging.info(f"Loading PCA model from: {model_path}")
            with open(model_path, 'rb') as f:
                self.pca_data = pickle.load(f)
                
            logging.info(f"PCA model loaded successfully (created on {self.pca_data.get('timestamp', 'unknown date')})")
            
            # Log variance explained
            if 'variance_explained' in self.pca_data:
                cumulative_var = np.cumsum(self.pca_data['variance_explained'])
                logging.info(f"PCA model explains {cumulative_var[-1]*100:.1f}% of variance using {len(cumulative_var)} components")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading PCA model: {str(e)}")
            return False

    def find_latest_pca_model(self):
        """Find the most recent PCA model file"""
        models_dir = self.base_dir / 'models'
        
        if not models_dir.exists():
            logging.error(f"Models directory not found: {models_dir}")
            return None
        
        # Find all PCA model files
        model_files = list(models_dir.glob('pca_model_*.pkl'))
        
        if not model_files:
            logging.error(f"No PCA model files found in {models_dir}")
            return None
        
        # Sort by modification time (most recent first)
        latest_model = max(model_files, key=os.path.getmtime)
        return latest_model
            
    def fetch_data(self):
        """Fetch historical data for clustering analysis"""
        try:
            # Access the bitget client correctly from the config
            if not hasattr(self, 'config') or self.config is None:
                raise ValueError("Configuration not provided")
            
            if 'bitget_client' not in self.config:
                raise ValueError("Bitget client not provided in config")
            
            # Use a direct approach that works with daily timeframes
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
            
            logging.info(f"Fetching data from {start_date} for {self.config['symbol']}")
            
            # Get the client from config and access correct properties
            data = self.config['bitget_client'].fetch_ohlcv(
                symbol=self.config['symbol'],           # Access symbol from config
                timeframe=self.config['timeframe'],     # Access timeframe from config
                start_time=start_date
            )
            
            logging.info(f"Fetched {len(data)} candles for {self.config['symbol']}")
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise
    
    def transform_with_pca(self, data):
        """Transform market data using the loaded PCA model"""
        try:
            if self.pca_data is None:
                raise ValueError("PCA model not loaded. Call load_pca_model first.")
            
            # Calculate features for the data
            features_df = calculate_all_features(data)
            
            # Select only the features used in the PCA model
            selected_features = self.pca_data['selected_features']
            X = features_df[selected_features].values
            
            # Standardize using the saved scaler
            X_scaled = self.pca_data['scaler'].transform(X)
            
            # Transform to principal components using the already trained model
            pc_values = self.pca_data['pca_model'].transform(X_scaled)
            
            # Create DataFrame with principal components
            pc_df = pd.DataFrame(
                pc_values, 
                columns=[f'PC{i+1}' for i in range(pc_values.shape[1])],
                index=features_df.index
            )
            
            # Ensure we have a timestamp column for later processing
            # Copy timestamp from index
            pc_df['timestamp'] = pc_df.index
            
            return pc_df
            
        except Exception as e:
            logging.error(f"Error transforming data with PCA: {str(e)}")
            raise
    
    def determine_optimal_clusters(self, pc_df, max_clusters=15):
        """Determine the optimal number of clusters using multiple methods"""
        # Extract only the PC columns (not timestamp)
        X = pc_df.drop('timestamp', axis=1).values
        
        # Store metrics for different k values
        results = []
        
        # Try different numbers of clusters
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Calculate validation metrics
            silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
            calinski = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0
            inertia = kmeans.inertia_
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'inertia': inertia
            })
            
            logging.info(f"K={k}: Silhouette={silhouette:.4f}, CH={calinski:.2f}, Inertia={inertia:.2f}")
        
        # Convert to DataFrame for easy analysis
        metrics_df = pd.DataFrame(results)
        
        # Plot the metrics
        plt.figure(figsize=(18, 6))
        
        # Silhouette score (higher is better)
        plt.subplot(1, 3, 1)
        plt.plot(metrics_df['k'], metrics_df['silhouette'], 'o-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score (higher is better)')
        plt.grid(True)
        
        # Calinski-Harabasz score (higher is better)
        plt.subplot(1, 3, 2)
        plt.plot(metrics_df['k'], metrics_df['calinski_harabasz'], 'o-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Score (higher is better)')
        plt.grid(True)
        
        # Inertia / Elbow method (look for the elbow)
        plt.subplot(1, 3, 3)
        plt.plot(metrics_df['k'], metrics_df['inertia'], 'o-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method (look for the bend)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'analysis' / f'cluster_metrics_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
        
        # Find best k based on silhouette score (simple approach)
        best_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        logging.info(f"Suggested optimal number of clusters (based on silhouette): {best_k}")
        
        return best_k, metrics_df
    
    def perform_clustering(self, pc_df, n_clusters=None):
        """Perform clustering on the principal components"""
        try:
            # If number of clusters not provided, determine optimal
            if n_clusters is None:
                n_clusters, _ = self.determine_optimal_clusters(pc_df)
            
            # Extract only the PC columns (not timestamp)
            X = pc_df.drop('timestamp', axis=1).values
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Add cluster labels to the DataFrame
            pc_df['cluster'] = cluster_labels
            
            # Store the model
            self.kmeans_model = kmeans
            
            # Calculate and store cluster statistics
            self.calculate_cluster_statistics(pc_df)
            
            # Visualize clusters
            self.visualize_clusters(pc_df)
            
            return pc_df
            
        except Exception as e:
            logging.error(f"Error performing clustering: {str(e)}")
            raise
    
    def calculate_cluster_statistics(self, pc_df):
        """Calculate statistics for each cluster"""
        # Group by cluster and calculate basic statistics
        cluster_stats = {}
        
        for cluster_id in sorted(pc_df['cluster'].unique()):
            cluster_data = pc_df[pc_df['cluster'] == cluster_id]
            
            # Calculate statistics for this cluster
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(pc_df) * 100,
                'center': {
                    col: cluster_data[col].mean() 
                    for col in cluster_data.columns if col not in ['timestamp', 'cluster']
                }
            }
            
            cluster_stats[int(cluster_id)] = stats
            
            # Log some info about this cluster
            logging.info(f"Cluster {cluster_id}: {stats['size']} samples ({stats['percentage']:.1f}% of total)")
            center_formatted = ", ".join([f"{col}={val:.2f}" for col, val in stats['center'].items()])
            logging.info(f"  Center: {center_formatted}")
        
        # Store cluster statistics
        self.cluster_stats = cluster_stats
        return cluster_stats
    
    def visualize_clusters(self, pc_df):
        """Create visualizations of the clusters"""
        try:
            # 2D scatter plot of PC1 vs PC2 with cluster coloring
            plt.figure(figsize=(12, 10))
            
            # Create scatter plot with cluster coloring
            scatter = plt.scatter(
                pc_df['PC1'], 
                pc_df['PC2'], 
                c=pc_df['cluster'], 
                cmap='viridis', 
                alpha=0.7,
                s=50
            )
            
            # Add cluster centers
            centers = self.kmeans_model.cluster_centers_
            plt.scatter(
                centers[:, 0], 
                centers[:, 1], 
                c='red', 
                marker='X', 
                s=200, 
                label='Cluster Centers'
            )
            
            # Add legend and labels
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('Principal Component 1 (Market Direction)')
            plt.ylabel('Principal Component 2 (Price Structure)')
            plt.title('Market Regime Clusters in PC Space')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(self.base_dir / 'analysis' / f'clusters_2d_{datetime.now().strftime("%Y%m%d")}.png')
            plt.close()
            
            # 3D scatter plot if we have at least 3 PCs
            if 'PC3' in pc_df.columns:
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    fig = plt.figure(figsize=(14, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Create 3D scatter plot
                    scatter = ax.scatter(
                        pc_df['PC1'], 
                        pc_df['PC2'], 
                        pc_df['PC3'],
                        c=pc_df['cluster'], 
                        cmap='viridis', 
                        alpha=0.7,
                        s=50
                    )
                    
                    # Add cluster centers
                    ax.scatter(
                        centers[:, 0], 
                        centers[:, 1], 
                        centers[:, 2],
                        c='red', 
                        marker='X', 
                        s=200, 
                        label='Cluster Centers'
                    )
                    
                    # Add labels
                    ax.set_xlabel('PC1 (Market Direction)')
                    ax.set_ylabel('PC2 (Price Structure)')
                    ax.set_zlabel('PC3 (Volatility)')
                    plt.title('Market Regime Clusters in 3D PC Space')
                    
                    # Add colorbar
                    plt.colorbar(scatter, label='Cluster')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(self.base_dir / 'analysis' / f'clusters_3d_{datetime.now().strftime("%Y%m%d")}.png')
                    plt.close()
                except Exception as e:
                    logging.warning(f"Could not create 3D cluster plot: {str(e)}")
            
            # Create a heatmap of cluster centers
            center_df = pd.DataFrame(
                self.kmeans_model.cluster_centers_, 
                columns=[col for col in pc_df.columns if col not in ['timestamp', 'cluster']]
            )
            center_df.index = [f'Cluster {i}' for i in range(len(center_df))]
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(center_df, annot=True, cmap='coolwarm', center=0)
            plt.title('Cluster Centers in Principal Component Space')
            plt.tight_layout()
            plt.savefig(self.base_dir / 'analysis' / f'cluster_centers_{datetime.now().strftime("%Y%m%d")}.png')
            plt.close()
            
            logging.info(f"Cluster visualizations saved to {self.base_dir / 'analysis'}")
            
        except Exception as e:
            logging.error(f"Error visualizing clusters: {str(e)}")
    
    def interpret_clusters(self):
        """Provide interpretation of each cluster"""
        if self.cluster_stats is None:
            logging.warning("No cluster statistics available. Run perform_clustering first.")
            return
        
        # Interpret each cluster based on its center in PC space
        interpretations = {}
        
        # If we have the PCA loadings, use them to interpret
        component_meanings = {
            'PC1': "Market Direction & Trend",
            'PC2': "Price Structure & Support",
            'PC3': "Volatility Regime",
            'PC4': "Short-term Volume Surge",
            'PC5': "OBV Dominance"
        }
        
        for cluster_id, stats in self.cluster_stats.items():
            center = stats['center']
            
            # Start with basic size info
            interpretation = [
                f"Size: {stats['size']} samples ({stats['percentage']:.1f}% of data)"
            ]
            
            # Interpret based on each principal component
            for pc_name, value in center.items():
                if pc_name in component_meanings:
                    meaning = component_meanings[pc_name]
                    strength = abs(value)
                    direction = "high" if value > 0 else "low"
                    
                    if strength > 1.5:
                        intensity = "very strong"
                    elif strength > 0.8:
                        intensity = "strong"
                    elif strength > 0.3:
                        intensity = "moderate"
                    else:
                        intensity = "weak"
                    
                    interpretation.append(f"{pc_name} ({meaning}): {intensity} {direction} ({value:.2f})")
            
            # Add market regime characterization based on primary components
            if 'PC1' in center and 'PC3' in center:
                # Direction (PC1) and Volatility (PC3)
                pc1, pc3 = center['PC1'], center['PC3']
                
                # Market regime characterization
                if pc1 < -0.8 and pc3 > 0.8:
                    regime = "Bullish Breakout"
                elif pc1 < -0.8 and pc3 < -0.8:
                    regime = "Stable Uptrend"
                elif pc1 > 0.8 and pc3 > 0.8:
                    regime = "Bearish Breakdown"
                elif pc1 > 0.8 and pc3 < -0.8:
                    regime = "Stable Downtrend"
                elif abs(pc1) < 0.3 and pc3 > 0.8:
                    regime = "Volatile Range"
                elif abs(pc1) < 0.3 and pc3 < -0.3:
                    regime = "Quiet Range"
                else:
                    regime = "Transitional"
                
                interpretation.append(f"Market Regime: {regime}")
            
            # Suggested trading approach
            if 'PC1' in center:
                pc1 = center['PC1']
                if pc1 < -0.8:
                    trading = "Bullish bias, look for buying opportunities"
                elif pc1 > 0.8:
                    trading = "Bearish bias, look for selling opportunities"
                else:
                    trading = "Neutral bias, range trading opportunities"
                
                interpretation.append(f"Trading Approach: {trading}")
            
            interpretations[cluster_id] = interpretation
        
        # Log the interpretations
        logging.info("\n=== CLUSTER INTERPRETATIONS ===")
        for cluster_id, interp in interpretations.items():
            logging.info(f"\nCluster {cluster_id}:")
            for line in interp:
                logging.info(f"  - {line}")
        
        return interpretations
    
    def save_clustering_model(self):
        """Save the clustering model for later use"""
        try:
            if self.kmeans_model is None or self.pca_data is None:
                raise ValueError("Both PCA and K-means models must be available before saving")
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d")
            
            # Prepare data to save
            save_data = {
                'kmeans_model': self.kmeans_model,
                'pca_data': self.pca_data,
                'cluster_stats': self.cluster_stats,
                'timestamp': timestamp
            }
            
            # Save with pickle
            save_path = self.base_dir / 'models' / f'market_clustering_{timestamp}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logging.info(f"Clustering model saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error saving clustering model: {str(e)}")
            raise
    
    def load_clustering_model(self, path=None):
        """Load a previously saved clustering model"""
        try:
            # If no path provided, find the latest
            if path is None:
                models_dir = self.base_dir / 'models'
                model_files = list(models_dir.glob('market_clustering_*.pkl'))
                
                if not model_files:
                    logging.error("No clustering model files found")
                    return False
                
                path = max(model_files, key=os.path.getmtime)
            
            # Load the model
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract components
            self.kmeans_model = model_data['kmeans_model']
            self.pca_data = model_data['pca_data']
            self.cluster_stats = model_data['cluster_stats']
            
            logging.info(f"Clustering model loaded from {path}")
            logging.info(f"Model created on: {model_data.get('timestamp', 'unknown')}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading clustering model: {str(e)}")
            return False
    
    def classify_new_data(self, new_data):
        """Classify new market data into clusters"""
        try:
            if self.kmeans_model is None or self.pca_data is None:
                raise ValueError("Both PCA and K-means models must be loaded before classification")
            
            # Transform to PC space using PCA
            pc_df = self.transform_with_pca(new_data)
            
            # Predict cluster for each sample
            X = pc_df.drop('timestamp', axis=1).values
            cluster_labels = self.kmeans_model.predict(X)
            
            # Add to the DataFrame
            pc_df['cluster'] = cluster_labels
            
            # For the most recent data point, get its cluster
            latest_cluster = cluster_labels[-1]
            logging.info(f"Latest market data classified as Cluster {latest_cluster}")
            
            # Get interpretation for this cluster
            if self.cluster_stats and latest_cluster in self.cluster_stats:
                logging.info(f"Cluster {latest_cluster} characteristics:")
                center = self.cluster_stats[latest_cluster]['center']
                for pc, value in center.items():
                    logging.info(f"  {pc}: {value:.2f}")
            
            return latest_cluster, pc_df
            
        except Exception as e:
            logging.error(f"Error classifying new data: {str(e)}")
            raise


# Main execution if run directly
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Create base directory
    base_dir = Path(__file__).resolve().parents[1]
    
    try:
        # Load configuration
        key_path = base_dir / 'config' / 'config.json'
        with open(key_path, "r") as f:
            api_setup = json.load(f)['bitget']
        
        # Initialize BitgetFutures client
        from utils.bitget_futures import BitgetFutures
        bitget_client = BitgetFutures(api_setup)
        
        # Create config for clustering
        config = {
            'symbol': 'ETH/USDT:USDT',
            'timeframe': '1d',
            'limit': 365,
            'bitget_client': bitget_client
        }
        
        logging.info("Configuration loaded successfully")
        
        # Initialize the clustering model
        clustering = MarketClustering(config)
        
        # First, load an existing clustering model if available
        if clustering.load_clustering_model():
            logging.info("Found existing clustering model, using it for analysis")
            
            # Get recent market data
            # Fetch the most recent data - just enough to classify current state
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            recent_data = bitget_client.fetch_ohlcv(
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                start_time=start_date
            )
            
            # Classify the current market regime
            latest_cluster, pc_df = clustering.classify_new_data(recent_data)
            
            logging.info(f"Current market is in Cluster {latest_cluster}")
            
            # Get interpretation
            interpretations = clustering.interpret_clusters()
            
            # Display results
            print("\n" + "="*50)
            print("CURRENT MARKET REGIME ANALYSIS")
            print("="*50)
            print(f"\nCurrent market is in CLUSTER {latest_cluster}")
            
            if interpretations and latest_cluster in interpretations:
                print("\nCharacteristics of this regime:")
                for line in interpretations[latest_cluster]:
                    print(f"  • {line}")
            
            # Show trading approach
            print("\nTrading approach for this regime:")
            for line in interpretations[latest_cluster]:
                if line.startswith("Trading Approach:"):
                    print(f"  • {line}")
            
            # Print the most recent PC values to understand exact position
            print("\nLatest principal component values:")
            latest_pc = pc_df.iloc[-1].drop('cluster')
            for pc, value in latest_pc.items():
                print(f"  • {pc}: {value:.3f}")
                
            print("\n" + "="*50)
            
        else:
            # If no model exists, create one from scratch with full pipeline
            logging.info("No existing clustering model found. Running full analysis...")
            
            # Load the PCA model
            if clustering.load_pca_model():
                # Fetch market data
                data = clustering.fetch_data()
                
                # Transform to PCA space
                pc_df = clustering.transform_with_pca(data)
                logging.info(f"Transformed data to {pc_df.shape[1]-1} principal components")
                
                # Find optimal number of clusters
                optimal_k, metrics = clustering.determine_optimal_clusters(pc_df)
                
                # Perform clustering
                clustered_df = clustering.perform_clustering(pc_df, n_clusters=optimal_k)
                logging.info(f"Performed clustering with {optimal_k} clusters")
                
                # Interpret clusters
                interpretations = clustering.interpret_clusters()
                
                # Save the clustering model
                model_path = clustering.save_clustering_model()
                logging.info(f"Clustering model saved to: {model_path}")
                
                # Example of classifying the most recent data
                latest_data = data.tail(30)  # Last 30 candles
                latest_cluster, _ = clustering.classify_new_data(latest_data)
                
                print("\n=== CURRENT MARKET REGIME ===")
                print(f"Current market is in Cluster {latest_cluster}")
                if interpretations and latest_cluster in interpretations:
                    for line in interpretations[latest_cluster]:
                        print(f"  • {line}")
                
                # Show recommendation based on current cluster
                print("\n=== TRADING RECOMMENDATION ===")
                if interpretations and latest_cluster in interpretations:
                    for line in interpretations[latest_cluster]:
                        if line.startswith("Trading Approach:"):
                            print(f"{line}")
            
            else:
                logging.error("Failed to load PCA model. Please run pca_analyzer.py first.")
            
    except Exception as e:
        logging.error(f"Error in market clustering: {str(e)}")
        import traceback
        traceback.print_exc() 