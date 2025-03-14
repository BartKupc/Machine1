import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import pickle
import os

# Add this to properly import from parent directory
import sys
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features

class PCAAnalyzer:
    def __init__(self, config=None):
        """Initialize the PCA analyzer"""
        self.base_dir = base_dir
        self.config = config
        
        # Set default parameters
        self.n_components = 5  # Default number of components to keep
        self.pca_model = None
        self.scaler = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Ensure models directory exists
        (self.base_dir / 'models').mkdir(parents=True, exist_ok=True)
        
        # Pre-selected important features (from initial run)
        self.selected_features = [
            'macd_diff',
            'sma50',
            'stoch_d',
            'obv',
            'stoch_k',
            'senkou_span_a',
            'price_change_3',
            'price_ema9_ratio',
            'roc_5',
            'price_change_5'
        ]
        
    def fetch_data(self):
        """Fetch historical data for PCA analysis"""
        try:
            if not hasattr(self, 'config') or self.config is None:
                raise ValueError("Configuration not provided")
                
            if 'bitget_client' not in self.config:
                raise ValueError("Bitget client not provided in config")
            
            # Use the existing start_time parameter in fetch_ohlcv instead of fetch_recent_ohlcv
            # This handles daily timeframes correctly
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
            
            # Use the correct fetch_ohlcv method with start_time
            data = self.config['bitget_client'].fetch_ohlcv(
                symbol=self.config['symbol'],
                timeframe=self.config['timeframe'],
                start_time=start_date
            )
            
            logging.info(f"Fetched {len(data)} candles for {self.config['symbol']}")
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise
            
    def perform_pca(self, data):
        """Perform PCA analysis on calculated features"""
        try:
            # Calculate features if needed
            if isinstance(data, pd.DataFrame) and 'close' in data.columns:
                features_df = calculate_all_features(data.copy())
            else:
                features_df = data.copy()
            
            # Extract selected features
            feature_data = features_df[self.selected_features].copy()
            
            # Drop rows with NaN values
            feature_data = feature_data.dropna()
            
            # Store the data for visualization later
            self.feature_data = feature_data
            
            # Standardize the data
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Determine optimal number of components
            if 'pca_variance_threshold' in self.config:
                variance_threshold = self.config['pca_variance_threshold']
            else:
                variance_threshold = 0.85  # Default: capture 85% of variance
            
            # Initialize PCA with max components
            full_pca = PCA()
            full_pca.fit(scaled_data)
            
            # Determine number of components needed to explain variance_threshold of variance
            explained_variance = full_pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # Perform PCA with optimal components
            self.pca_model = PCA(n_components=n_components)
            principal_components = self.pca_model.fit_transform(scaled_data)
            
            # Store principal components for later
            self.principal_components = principal_components
            
            # Analyze component loadings
            loadings = self.pca_model.components_
            
            # Calculate feature importance
            feature_importance = np.sum(np.abs(loadings), axis=0)
            
            # Normalize feature importance
            feature_importance = feature_importance / np.sum(feature_importance)
            
            # Create sorted list of (feature, importance) tuples
            sorted_features = sorted(
                zip(self.selected_features, feature_importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create visualizations
            self.visualize_pca(principal_components, data)
            self.visualize_feature_importance(sorted_features)
            self.analyze_component_loadings(loadings, n_components)
            
            # Return results
            results = {
                'pca_model': self.pca_model,
                'scaler': self.scaler,
                'principal_components': principal_components,
                'selected_features': self.selected_features,
                'sorted_features': sorted_features,
                'variance_explained': explained_variance,
                'n_components': n_components
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in PCA analysis: {str(e)}")
            raise
    
    def visualize_pca(self, principal_components, data):
        """Create PCA visualization plots"""
        try:
            # Create figure directory if it doesn't exist
            figures_dir = self.base_dir / 'analysis' / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # 2D Scatter plot of first two principal components
            plt.figure(figsize=(12, 8))
            
            # Create a colormap based on price changes
            price_changes = data['close'].pct_change(5).values[5:]  # 5-day returns
            cmap = plt.cm.coolwarm
            
            # Plotting the points
            scatter = plt.scatter(
                principal_components[5:, 0], 
                principal_components[5:, 1],
                c=price_changes, 
                cmap=cmap,
                alpha=0.7, 
                s=30
            )
            
            plt.colorbar(scatter, label='5-Day Return')
            plt.title('PCA: First Two Principal Components')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
            
            # Save the figure
            plt.savefig(figures_dir / 'pca_2d_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3D Scatter plot if we have at least 3 components
            if principal_components.shape[1] >= 3:
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                scatter = ax.scatter(
                    principal_components[5:, 0],
                    principal_components[5:, 1],
                    principal_components[5:, 2],
                    c=price_changes,
                    cmap=cmap,
                    alpha=0.7,
                    s=30
                )
                
                plt.colorbar(scatter, label='5-Day Return')
                ax.set_title('PCA: First Three Principal Components')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_zlabel('Principal Component 3')
                
                # Save the figure
                plt.savefig(figures_dir / 'pca_3d_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            logging.info(f"PCA visualizations saved to {figures_dir}")
            
        except Exception as e:
            logging.error(f"Error creating PCA visualizations: {str(e)}")
            
    def visualize_feature_importance(self, sorted_features):
        """Visualize feature importance from PCA"""
        try:
            # Create figure directory if it doesn't exist
            figures_dir = self.base_dir / 'analysis' / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract feature names and importance scores
            feature_names = [feature for feature, _ in sorted_features]
            importance_scores = [score for _, score in sorted_features]
            
            # Create horizontal bar chart
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(feature_names)), importance_scores, align='center')
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Relative Importance')
            plt.title('Feature Importance Based on PCA Component Loadings')
            plt.tight_layout()
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                         f'{width:.4f}', ha='left', va='center')
            
            # Save the figure
            plt.savefig(figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Feature importance visualization saved to {figures_dir}")
            
        except Exception as e:
            logging.error(f"Error creating feature importance visualization: {str(e)}")
    
    def analyze_component_loadings(self, loadings, n_components):
        """Analyze and visualize component loadings"""
        try:
            # Create figure directory if it doesn't exist
            figures_dir = self.base_dir / 'analysis' / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare loadings for visualization
            loadings_df = pd.DataFrame(
                loadings[:n_components].T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=self.selected_features
            )
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('PCA Component Loadings')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(figures_dir / 'component_loadings.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log top contributing features for each component
            logging.info("\nTop contributing features by component:")
            for i in range(n_components):
                component_loadings = loadings[i]
                sorted_idx = np.argsort(np.abs(component_loadings))[::-1]
                
                logging.info(f"\nPC{i+1} top features:")
                for idx in sorted_idx[:5]:  # Top 5 features
                    feature = self.selected_features[idx]
                    loading = component_loadings[idx]
                    logging.info(f"  {feature}: {loading:.4f}")
            
        except Exception as e:
            logging.error(f"Error analyzing component loadings: {str(e)}")
    
    def save_pca_model(self):
        """Save the PCA model and related data for later use"""
        try:
            if self.pca_model is None:
                raise ValueError("PCA model not available. Run perform_pca first.")
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d")
            
            # Prepare data to save
            save_data = {
                'pca_model': self.pca_model,
                'scaler': self.scaler,
                'selected_features': self.selected_features,
                'timestamp': timestamp,
                'symbol': self.config.get('symbol', 'unknown'),
                'timeframe': self.config.get('timeframe', 'unknown'),
                'variance_explained': self.pca_model.explained_variance_ratio_
            }
            
            # Add principal components if available
            if hasattr(self, 'principal_components'):
                save_data['principal_components'] = self.principal_components
            
            # Save with pickle
            save_path = self.base_dir / 'models' / f'pca_model_{timestamp}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logging.info(f"PCA model saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error saving PCA model: {str(e)}")
            raise

    def interpret_components(self, loadings, n_components=None):
        """Provide detailed interpretation of what each PCA component represents"""
        if n_components is None:
            n_components = min(5, loadings.shape[0])  # Default to top 5 or fewer
        
        # Create a nice formatted display of the components
        print("\n" + "="*50)
        print("PCA COMPONENT INTERPRETATION")
        print("="*50)
        
        for i in range(n_components):
            # Get loadings for this component
            component_loadings = loadings[i]
            
            # Get sorted indices (by absolute magnitude)
            sorted_idx = np.argsort(np.abs(component_loadings))[::-1]
            
            # Print component header
            variance = self.pca_model.explained_variance_ratio_[i] * 100
            print(f"\n\nCOMPONENT {i+1} - {variance:.2f}% of variance")
            print("-" * 40)
            
            # Print positive and negative loadings separately
            pos_loadings = [(self.selected_features[j], component_loadings[j]) 
                            for j in sorted_idx if component_loadings[j] > 0.2]
            neg_loadings = [(self.selected_features[j], component_loadings[j]) 
                            for j in sorted_idx if component_loadings[j] < -0.2]
            
            # Print positive loadings (features that increase when component increases)
            print("HIGH values in this component mean:")
            if pos_loadings:
                for feature, loading in pos_loadings:
                    print(f"  • HIGH {feature} ({loading:.3f})")
            else:
                print("  • No strong positive indicators")
                
            # Print negative loadings (features that decrease when component increases)
            print("\nLOW values in this component mean:")
            if neg_loadings:
                for feature, loading in neg_loadings:
                    print(f"  • LOW {feature} ({loading:.3f})")
            else:
                print("  • No strong negative indicators")
            
            # Add interpretation based on the loadings
            print("\nPossible interpretation:")
            interpretation = self._interpret_component(i, pos_loadings, neg_loadings)
            print(f"  {interpretation}")
        
        print("\n" + "="*50)
        
    def _interpret_component(self, component_idx, pos_loadings, neg_loadings):
        """Generate interpretation text for a component based on its loadings"""
        # Get feature names for easier pattern recognition
        pos_features = [f[0] for f in pos_loadings]
        neg_features = [f[0] for f in neg_loadings]
        
        # Component 1 - usually represents overall market trend/momentum
        if component_idx == 0:
            if any(f in pos_features for f in ['macd_diff', 'roc_5', 'price_change_5']):
                return "BULLISH MOMENTUM - This component likely represents positive market momentum with rising prices"
            if any(f in neg_features for f in ['macd_diff', 'roc_5', 'price_change_5']):
                return "BEARISH MOMENTUM - This component likely represents negative market momentum with falling prices"
            return "MARKET TREND - This component appears to capture the overall market direction"
        
        # Component 2 - often represents oscillation/volatility
        if component_idx == 1:
            if any(f in pos_features for f in ['stoch_k', 'stoch_d']):
                return "OVERBOUGHT CONDITIONS - High values of oscillators suggest potential reversal from highs"
            if any(f in neg_features for f in ['stoch_k', 'stoch_d']):
                return "OVERSOLD CONDITIONS - Low values of oscillators suggest potential reversal from lows"
            if 'bb_width' in pos_features or 'atr' in pos_features:
                return "HIGH VOLATILITY - This component appears to capture market volatility"
            return "MARKET OSCILLATION - This component appears to capture market cycles or reversals"
        
        # Component 3 - often represents divergence between indicators
        if component_idx == 2:
            return "INDICATOR DIVERGENCE - This component may represent divergence between price action and indicators"
        
        # Component 4 - often volume or liquidity related
        if component_idx == 3:
            if 'obv' in pos_features:
                return "STRONG VOLUME CONFIRMATION - Volume is confirming price movement"
            if 'obv' in neg_features:
                return "VOLUME DIVERGENCE - Volume is not confirming price movement"
            return "VOLUME/LIQUIDITY FACTOR - This component appears related to trading volume patterns"
        
        # Default response for other components
        return "MARKET FACTOR - This component represents a significant market pattern"

# Main execution section
if __name__ == "__main__":
    # Load configuration directly from config.json
    try:
        key_path = base_dir / 'config' / 'config.json'
        with open(key_path, "r") as f:
            api_setup = json.load(f)['bitget']
        
        # Initialize BitgetFutures client using the same method as in live_moment2.py
        bitget_client = BitgetFutures(api_setup)
        
        # Create config for analyzer - updated for daily timeframe
        config = {
            'symbol': 'ETH/USDT:USDT',
            'timeframe': '1d',         # Daily timeframe
            'limit': 365,              # Get about a year of daily data
            'pca_variance_threshold': 0.85,
            'bitget_client': bitget_client
        }
        
        logging.info("Starting PCA analysis with daily timeframe and pre-selected features")
        
        # Initialize and run PCA analysis
        pca = PCAAnalyzer(config)
        data = pca.fetch_data()
        pca_results = pca.perform_pca(data)
        
        # Save the model for later use
        save_path = pca.save_pca_model()
        logging.info(f"PCA model saved to: {save_path}")
        
        # Show explained variance
        explained_variance = pca_results['pca_model'].explained_variance_ratio_
        cum_variance = np.cumsum(explained_variance)
        print("\nExplained variance by component:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cum_variance)):
            print(f"PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")
        
        # Call the component interpretation directly
        pca.interpret_components(pca_results['pca_model'].components_, pca_results['n_components'])
    
    except Exception as e:
        logging.error(f"Error in PCA analysis: {str(e)}")
        import traceback
        traceback.print_exc()