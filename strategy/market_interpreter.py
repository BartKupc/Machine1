import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import io
import os
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import tarfile
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add parent directory to path for imports
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.feature_calculator import calculate_all_features
from utils.bitget_futures import BitgetFutures


class MarketConditionInterpreter:
    """
    Interprets current market conditions based on PCA model
    and generates trading signals (long/short recommendations).
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the interpreter with PCA model components.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved PCA model components. If None, will load from latest S3 data.
        """
        self.bucket_name = 'sagemaker-eu-west-1-688567281415'
        self.s3_client = boto3.client('s3')
        
        # Load PCA components
        if model_path:
            self.load_model_from_file(model_path)
        else:
            self.load_model_from_s3()
        
        # Define parameters for trading signals
        self.momentum_threshold = 1.0  # Standard deviations from mean
        self.rsi_threshold = 1.0  # Standard deviations from mean
        self.trend_threshold = 1.0  # Standard deviations from mean for ADX (PC3)
        
        # Load historical data to calculate statistics for the PCA components
        self.historical_data = self.load_historical_transformed_data()
        self.pc1_mean = self.historical_data['PC1'].mean()
        self.pc1_std = self.historical_data['PC1'].std()
        self.pc2_mean = self.historical_data['PC2'].mean()
        self.pc2_std = self.historical_data['PC2'].std()
        
        # Add PC3 statistics (trend component)
        if 'PC3' in self.historical_data.columns:
            self.pc3_mean = self.historical_data['PC3'].mean()
            self.pc3_std = self.historical_data['PC3'].std()
        else:
            self.pc3_mean = 0
            self.pc3_std = 1
            logging.warning("PC3 not found in historical data. Using default values.")

    def load_model_from_file(self, model_path):
        """Load PCA model components from a file"""
        try:
            model_data = np.load(model_path, allow_pickle=True)
            self.pca_mean = model_data['mean']
            self.pca_components = model_data['components']
            self.feature_names = model_data['feature_names']
            logging.info(f"Loaded PCA model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model from file: {str(e)}")
            raise

    def load_model_from_s3(self):
        """Load the PCA model components from a hardcoded S3 path using the approach from interpret_pca_results.py."""
        try:
            # Hardcoded model key
            model_key = 'pca_output/pca-2025-03-14-09-21-21-722/output/model.tar.gz'
            logging.info(f"Loading model from s3://{self.bucket_name}/{model_key}")
            
            # Download the model
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=model_key)
            tar_data = io.BytesIO(response['Body'].read())
            
            # Create a temporary directory for the model
            tmp_dir = 'tmp_model'
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Extract the tar.gz file
            tar_path = os.path.join(tmp_dir, 'model.tar.gz')
            with open(tar_path, 'wb') as f:
                f.write(tar_data.getvalue())
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=tmp_dir)
            
            logging.info(f"Extracted model to {tmp_dir}")
            
            # List the extracted files for debugging
            model_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file != 'model.tar.gz':  # Skip the original tar file
                        file_path = os.path.join(root, file)
                        logging.info(f"  - {file_path}")
                        model_files.append(file_path)
            
            # Now load the data and fit PCA with scikit-learn
            # This is similar to fit_pca_with_sklearn() in interpret_pca_results.py
            
            # Load latest data
            latest_data = self.load_latest_data()
            
            # Fit PCA
            n_components = 5  # Same as in original code
            logging.info(f"Fitting PCA with {n_components} components")
            pca = PCA(n_components=n_components)
            pca.fit(latest_data)
            
            # Extract components
            self.pca_mean = pca.mean_
            self.pca_components = pca.components_
            self.feature_names = latest_data.columns.tolist()
            
            logging.info(f"Fitted PCA model with {n_components} components")
            logging.info(f"Mean shape: {self.pca_mean.shape}")
            logging.info(f"Components shape: {self.pca_components.shape}")
            logging.info(f"Feature names: {self.feature_names}")
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(tmp_dir)
            logging.info(f"Cleaned up temporary model directory")
            
        except Exception as e:
            logging.error(f"Error loading model from S3: {str(e)}")
            raise

    def save_model_locally(self):
        """Save the PCA model components locally"""
        try:
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Create a timestamp-based filename
            now = datetime.now()
            model_path = model_dir / f"pca_model_{now.strftime('%Y%m%d_%H%M')}.npz"
            
            # Save the model components
            np.savez(
                model_path, 
                mean=self.pca_mean, 
                components=self.pca_components,
                feature_names=self.feature_names
            )
            
            logging.info(f"Saved PCA model to {model_path}")
            
        except Exception as e:
            logging.error(f"Error saving model locally: {str(e)}")

    def load_latest_data(self):
        """Load the latest data file from S3"""
        # List all objects in the bucket
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name
        )
        
        # Filter for CSV files and find the most recent PCA_*.csv file
        csv_files = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv') and 'PCA_' in key:
                csv_files.append((key, obj['LastModified']))
        
        if not csv_files:
            raise ValueError(f"No PCA data files found in s3://{self.bucket_name}")
        
        # Sort by last modified time to get the most recent
        latest_data_key = sorted(csv_files, key=lambda x: x[1], reverse=True)[0][0]
        logging.info(f"Found latest data file: {latest_data_key}")
        
        # Download the CSV file
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=latest_data_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        
        # First column is typically the index
        if 'timestamp' in df.columns or 'Unnamed: 0' in df.columns:
            index_col = 'timestamp' if 'timestamp' in df.columns else 'Unnamed: 0'
            df.set_index(index_col, inplace=True)
        
        logging.info(f"Loaded data with shape {df.shape}")
        return df

    def load_historical_transformed_data(self):
        """Load the historical transformed data from the PCA analysis"""
        try:
            # Try to load the historical transformed data from a local file
            if os.path.exists('pca_transformed_data.csv'):
                df = pd.read_csv('pca_transformed_data.csv')
                if 'Unnamed: 0' in df.columns:
                    df.set_index('Unnamed: 0', inplace=True)
                logging.info(f"Loaded historical transformed data with shape {df.shape}")
                return df
            else:
                # If the file doesn't exist, load the latest data and transform it
                latest_data = self.load_latest_data()
                transformed_df = self.transform_data(latest_data)
                logging.info(f"Created historical transformed data with shape {transformed_df.shape}")
                return transformed_df
        except Exception as e:
            logging.error(f"Error loading historical transformed data: {str(e)}")
            # If there's an error, create a default DataFrame
            return pd.DataFrame({'PC1': [0], 'PC2': [0]})

    def transform_data(self, data):
        """Transform data into PCA space"""
        # Ensure data has the same columns as the PCA model
        data = data[self.feature_names].copy()
        
        # Center the data
        centered_data = data.values - self.pca_mean
        
        # Project onto principal components
        transformed_data = np.dot(centered_data, self.pca_components.T)
        
        # Convert to DataFrame for easier handling
        transformed_df = pd.DataFrame(
            transformed_data,
            columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])],
            index=data.index
        )
        
        return transformed_df

    def get_current_market_data(self, symbol='ETH/USDT:USDT', timeframe='1h', candles_back=300):
        """
        Fetch current market data and calculate features.
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        timeframe : str
            Timeframe for the data (e.g., '1d', '4h', '1h')
        candles_back : int
            Number of candles to fetch
        
        Returns:
        --------
        DataFrame
            DataFrame with calculated features
        """
        try:
            # Load configuration
            key_path = base_dir / 'config' / 'config.json'
            import json
            with open(key_path, "r") as f:
                api_setup = json.load(f)['bitget']
            
            # Initialize BitgetFutures client
            bitget_client = BitgetFutures(api_setup)
            
            # Calculate start date based on number of candles and timeframe
            # Approximate days needed based on candles and timeframe
            hours_per_candle = {'1h': 1, '4h': 4, '1d': 24}[timeframe]
            days_needed = max(candles_back * hours_per_candle / 24 * 1.2, 2)  # Add 20% buffer and minimum 2 days
            
            start_date = (datetime.now() - timedelta(days=int(days_needed))).strftime('%Y-%m-%d')
            
            logging.info(f"Fetching {timeframe} data from {start_date} for {symbol} (targeting {candles_back} candles)")
            
            # Fetch data
            data = bitget_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date
            )
            
            logging.info(f"Fetched {len(data)} candles for {symbol}")
            
            # Calculate features
            features_df = self.process_market_data(data)
            
            return features_df
            
        except Exception as e:
            logging.error(f"Error fetching current market data: {str(e)}")
            raise

    def process_market_data(self, data):
        """
        Calculate features from raw market data and align with PCA features.
        
        Parameters:
        -----------
        data : list
            List of OHLCV data
        
        Returns:
        --------
        DataFrame
            DataFrame with the features required for the PCA model
        """
        try:
            # Calculate all available features
            features_df = calculate_all_features(data.copy())
            
            # Create DataFrame for OHLCV data
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamps to datetime if they aren't already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Calculate ema9_ratio if it's in the feature names
            if 'ema9_ratio' in self.feature_names and 'ema9' in features_df.columns:
                features_df['ema9_ratio'] = features_df['ema9'] / df['close'].values
                features_df['ema9_ratio'] = (features_df['ema9_ratio'] - 1) * 100
            
            # Apply the same transformations as in the PCA script
            
            # For volume_ratio_50
            if 'volume_ratio_50' in self.feature_names and 'volume_ratio_50' in features_df.columns:
                features_df['volume_ratio_50'] = np.sign(features_df['volume_ratio_50']) * np.sqrt(np.abs(features_df['volume_ratio_50']))
            
            # For supertrend
            if 'supertrend' in self.feature_names and 'supertrend' in features_df.columns:
                features_df['supertrend'] = features_df['supertrend'] / df['close'].values
            
            # For historical_volatility_30
            if 'historical_volatility_30' in self.feature_names and 'historical_volatility_30' in features_df.columns:
                mean_close = df['close'].mean()
                features_df['historical_volatility_30'] = features_df['historical_volatility_30'] / mean_close
            
            # For vwap_ratio
            if 'vwap_ratio' in self.feature_names and 'vwap_ratio' in features_df.columns:
                features_df['vwap_ratio'] = features_df['vwap_ratio']  # Already a ratio
            
            # For obv_ratio
            if 'obv_ratio' in self.feature_names and 'obv_ratio' in features_df.columns:
                # Replace infinities and NaNs with 1.0 before applying log1p
                obv_ratio = features_df['obv_ratio'].replace([np.inf, -np.inf, np.nan], 1.0)
                # Apply log1p safely
                features_df['obv_ratio'] = np.log1p(np.abs(obv_ratio)) * np.sign(obv_ratio)
                logging.info(f"Applied safe log transformation to obv_ratio")
            
            # Select only the required features and handle NaN values
            selected_features = [f for f in self.feature_names if f in features_df.columns]
            if len(selected_features) < len(self.feature_names):
                missing_features = set(self.feature_names) - set(selected_features)
                logging.warning(f"Missing features: {missing_features}")
            
            selected_df = features_df[selected_features].copy()
            selected_df = selected_df.fillna(0)
            
            # Skip initial rows where ATR might be zero
            if 'atr_ratio' in selected_df.columns:
                non_zero_indices = selected_df['atr_ratio'].ne(0)
                if non_zero_indices.any():
                    non_zero_atr_index = non_zero_indices.idxmax()
                    selected_df = selected_df.loc[non_zero_atr_index:].copy()
                else:
                    logging.warning("No non-zero ATR values found")
                
            return selected_df
            
        except Exception as e:
            logging.error(f"Error processing market data: {str(e)}")
            # Return a single row DataFrame with zeros as a fallback
            fallback_df = pd.DataFrame({feature: [0] for feature in self.feature_names})
            return fallback_df

    def classify_market_regime(self, pc1, pc2):
        """
        Classify the market regime based on PC1 and PC2.
        """
        if pc1 > self.pc1_mean and pc2 > self.pc2_mean:
            return "Strong Trending Market"
        elif pc1 > self.pc1_mean and pc2 < self.pc2_mean:
            return "Momentum Without Trend"
        elif pc1 < self.pc1_mean and pc2 < self.pc2_mean:
            return "Choppy/Noisy Market"
        else:
            return "Undefined"

    def generate_trade_signal(self, pc1, pc2, pc3):
        """
        Generate trade signals based on PC1, PC2, and PC3.
        """
        if pc1 > 0 and pc2 > self.pc2_mean:
            return "Long"
        elif pc1 < 0 and pc2 < self.pc2_mean:
            return "Short"
        elif pc3 > self.pc3_mean:
            return "Avoid Trading"
        else:
            return "Hold"

    def get_current_market_condition(self, data=None):
        """
        Get the current market condition based on the latest market data.
        
        Parameters:
        -----------
        data : DataFrame, optional
            DataFrame with calculated features. If None, fetches current data.
        
        Returns:
        --------
        dict
            Dictionary with market condition information
        """
        try:
            if data is None:
                data = self.get_current_market_data()
            
            # Get the last row of data
            latest_data = data.iloc[-1:].copy()
            
            # Transform to PCA space
            transformed_data = self.transform_data(latest_data)
            latest_pc1 = transformed_data['PC1'].iloc[0]
            latest_pc2 = transformed_data['PC2'].iloc[0]
            
            # Add PC3 (trend component)
            if 'PC3' in transformed_data.columns:
                latest_pc3 = transformed_data['PC3'].iloc[0]
            else:
                latest_pc3 = 0
            
            # Classify market regime
            market_regime = self.classify_market_regime(latest_pc1, latest_pc2)
            
            # Generate trade signal
            trade_signal = self.generate_trade_signal(latest_pc1, latest_pc2, latest_pc3)
            
            # Create the result dictionary
            result = {
                'market_regime': market_regime,
                'trade_signal': trade_signal,
                'pc1': latest_pc1,
                'pc2': latest_pc2,
                'pc3': latest_pc3,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting current market condition: {str(e)}")
            return {
                'market_regime': 'ERROR',
                'trade_signal': 'NEUTRAL',
                'error': str(e)
            }

    def main(self):
        """Main function to run the market condition interpreter"""
        try:
            # Get the current market condition
            result = self.get_current_market_condition()
            
            # Check if we have a valid result
            if 'error' in result:
                print(f"\nERROR: {result['error']}")
                print("See logs for more details.")
                return
            
            # Print a summary
            print("\nMARKET CONDITION SUMMARY:")
            print(f"- Market Regime: {result['market_regime']}")
            print(f"- Trade Signal: {result['trade_signal']}")
            print(f"- PC1: {result['pc1']:.2f}")
            print(f"- PC2: {result['pc2']:.2f}")
            print(f"- PC3: {result['pc3']:.2f}")
            
        except Exception as e:
            logging.error(f"Error in main function: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")
            print("See logs for more details.")

if __name__ == "__main__":
    # Initialize the interpreter without a local model path to use S3
    interpreter = MarketConditionInterpreter()
    
    # Run the main function
    interpreter.main() 