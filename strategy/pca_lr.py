import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
import matplotlib.pyplot as plt                   # For charts and visualizations
from IPython.display import Image                 # For displaying images in the notebook
from IPython.display import display               # For displaying outputs in the notebook
from datetime import datetime, timedelta
import sys                                        # For writing outputs to notebook
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import logging
import seaborn as sns
import pickle
import gzip
import urllib
import csv
import sagemaker
from sagemaker import PCA
from sagemaker.session import Session                              
from sagemaker import get_execution_role
from sklearn.preprocessing import StandardScaler
from sagemaker.remote_function import remote
import scipy.stats as stats


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add this to properly import from parent directory
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

# Now try importing
from utils.bitget_futures import BitgetFutures
from utils.feature_calculator import calculate_all_features


# Define settings outside the class to be used with standalone remote function
sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name="eu-west-1"))
settings = dict(
    sagemaker_session=sm_session,
    role = 'arn:aws:iam::688567281415:role/service-role/AmazonSageMaker-ExecutionRole-20240913T093672',
    instance_type="ml.m5.xlarge",
    dependencies='./requirements.txt'
)

# Define remote function outside the class
@remote(**settings)
def run_pca_remote(data_input, role, n_components, bucket_name):
    """Remote function to run PCA analysis on SageMaker"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Create a copy to avoid modifying the original
    df = data_input.copy()
    
    # Print feature statistics before scaling for debugging
    print("Feature statistics before scaling:")
    print(df.describe().transpose()[['mean', 'std', 'min', 'max']])
    
    # Use StandardScaler for all features
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(df))
    data_scaled.columns = df.columns
    data_scaled.index = df.index
    
    # Print statistics after scaling to verify
    print("\nFeature statistics after scaling:")
    print(data_scaled.describe().transpose()[['mean', 'std', 'min', 'max']])
    
    # Verify all features have similar ranges after scaling
    print("\nStandardized feature ranges:")
    for col in data_scaled.columns:
        col_range = data_scaled[col].max() - data_scaled[col].min()
        print(f"{col}: range = {col_range:.4f}")

    # Create a new session inside the remote environment
    remote_session = sagemaker.Session()
    
    pca_estimator = PCA(
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',       # Larger instance type
        num_components=n_components,
        sagemaker_session=remote_session,
        output_path=f"s3://{bucket_name}/pca_output",
    )

    train_data = data_scaled.values.astype('float32')
    pca_estimator.fit(pca_estimator.record_set(train_data))
    return "PCA complete with StandardScaler"


class SageMakerPCA:
    def __init__(self, config=None):
        """Initialize SageMaker PCA processor"""
        self.config = config
        
        # Define SageMaker settings
        self.sm_session = sm_session  # Use the one defined outside
        self.settings = settings      # Use the one defined outside
        
        # Store the role for PCA estimator
        self.role = self.settings['role']
        self.sagemaker_session = self.sm_session
        self.s3_client = boto3.client('s3')
        
        self.bucket_name = 'sagemaker-eu-west-1-688567281415'  # Replace with your actual bucket name
        
        # Extract necessary config items
        if config:
            self.symbol = config.get('symbol', 'ETH/USDT:USDT')
            self.timeframe = config.get('timeframe', '1h')
            self.bitget_client = config.get('bitget_client', None)
            self.n_components = config.get('n_components', 5)
            # Minimum number of periods needed for indicators to initialize
            self.warmup_period = config.get('warmup_period', 30)  # Default 30 days warmup
            # Days to analyze (for 1h timeframe)
            self.days_to_analyze = config.get('days_to_analyze', 30)  # Default to 30 days for 1h
            # Correlation threshold for feature selection
            self.correlation_threshold = config.get('correlation_threshold', 0.1)
            # Future return periods to analyze for correlation
            self.future_return_periods = config.get('future_return_periods', [1, 3, 5, 10])
        else:
            self.warmup_period = 30
            self.days_to_analyze = 30
            self.correlation_threshold = 0.1
            self.future_return_periods = [1, 3, 5, 10]
        
        # Define features to be used in our analysis (now we'll filter these based on correlation)
        self.selected_features = [
            # Momentum
            'macd_diff',
            'rsi',
            'stoch_k',
            'stoch_d',
            'stoch_diff',
            'cci',
            'williams_r',
            'stoch_rsi_k',
            'stoch_rsi_d',
            'roc_3',
            'roc_5',
            'roc_10',
            'roc_20',
            
            # Trend
            'adx',
            'supertrend',
            'bb_width',
            'tenkan_kijun_diff',

            # Volatility
            'atr_ratio',
            'historical_volatility_30',
            'bb_pct',

            # Volume
            'vwap_ratio',
            'obv_ratio',
            'mfi',
            'cmf',
            'volume_ratio_20',
            'volume_ratio_50',
            'ema9_ratio'
        ]
        
        # The filtered features will be stored here after correlation analysis
        self.filtered_features = []

    def perform_correlation_analysis(self, features_df, price_data):
        """
        Perform Spearman correlation analysis on features against future price returns
        and filter features based on correlation threshold.
        
        Args:
            features_df: DataFrame of calculated features
            price_data: Original price data with OHLCV values
            
        Returns:
            Filtered DataFrame with only the meaningful features
        """
        logging.info("Performing correlation analysis against future returns...")
        
        # Create a copy of features dataframe
        df_corr = features_df.copy()
        
        # Create a price dataframe from price_data
        if isinstance(price_data, pd.DataFrame):
            price_df = price_data.copy()
        else:
            price_df = pd.DataFrame(price_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Align indices between feature_df and price_df
        if len(df_corr) != len(price_df):
            logging.warning(f"Length mismatch between features ({len(df_corr)}) and price data ({len(price_df)})")
            min_len = min(len(df_corr), len(price_df))
            df_corr = df_corr.iloc[-min_len:].reset_index(drop=True)
            price_df = price_df.iloc[-min_len:].reset_index(drop=True)
        
        # Calculate future returns for different periods
        for period in self.future_return_periods:
            price_df[f'future_return_{period}'] = price_df['close'].pct_change(period).shift(-period)
        
        # Add future returns to feature dataframe
        for period in self.future_return_periods:
            df_corr[f'future_return_{period}'] = price_df[f'future_return_{period}']
        
        # Drop NaN values created by future return calculation
        df_corr = df_corr.dropna()
        
        # Calculate Spearman correlation for each feature against each future return period
        correlation_results = {}
        for feature in self.selected_features:
            if feature not in df_corr.columns:
                logging.warning(f"Feature {feature} not found in dataset, skipping...")
                continue
                
            # Calculate correlations for each future return period
            feature_correlations = []
            for period in self.future_return_periods:
                correlation, p_value = stats.spearmanr(
                    df_corr[feature], 
                    df_corr[f'future_return_{period}'],
                    nan_policy='omit'
                )
                feature_correlations.append((period, correlation, p_value))
            
            # Store the results
            correlation_results[feature] = feature_correlations
        
        # Log correlation results
        logging.info("Spearman correlation results:")
        for feature, correlations in correlation_results.items():
            for period, corr, p_value in correlations:
                logging.info(f"{feature} vs future_return_{period}: correlation={corr:.4f}, p-value={p_value:.4f}")
        
        # Filter features based on correlation threshold
        filtered_features = []
        for feature, correlations in correlation_results.items():
            # Check if any correlation exceeds threshold (in absolute value)
            for period, corr, p_value in correlations:
                if abs(corr) >= self.correlation_threshold and p_value <= 0.05:
                    filtered_features.append(feature)
                    logging.info(f"Selected feature {feature} with correlation {corr:.4f} for period {period}")
                    break  # No need to check other periods once we've selected the feature
        
        # Update the filtered_features attribute
        self.filtered_features = filtered_features
        
        # Create a correlation heatmap and save it to S3
        self.save_correlation_heatmap(df_corr, self.filtered_features)
        
        # Log the filtered features
        logging.info(f"Selected {len(filtered_features)} features based on correlation threshold {self.correlation_threshold}:")
        logging.info(", ".join(filtered_features))
        
        # Return the filtered dataframe
        return features_df[filtered_features]
    
    def save_correlation_heatmap(self, df_corr, filtered_features):
        """Save correlation heatmap to S3"""
        try:
            # Create correlation matrix for filtered features and future returns
            columns_to_plot = filtered_features.copy()
            for period in self.future_return_periods:
                columns_to_plot.append(f'future_return_{period}')
            
            corr_matrix = df_corr[columns_to_plot].corr(method='spearman')
            
            # Create the heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title(f'Spearman Correlation Heatmap - {self.symbol} ({self.timeframe})')
            plt.tight_layout()
            
            # Save to local file
            now = datetime.now()
            folder_name = now.strftime("%y_%m_%d")
            file_name = f"correlation_heatmap_{self.timeframe}_{now.strftime('%H_%M')}.png"
            local_path = file_name
            plt.savefig(local_path)
            plt.close()
            
            # Upload to S3
            s3_path = f"{folder_name}/{file_name}"
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_path
            )
            
            # Clean up local file
            os.remove(local_path)
            
            logging.info(f"Saved correlation heatmap to s3://{self.bucket_name}/{s3_path}")
            
        except Exception as e:
            logging.error(f"Error saving correlation heatmap: {str(e)}")
            # Continue execution even if heatmap save fails


    def fetch_data(self):
        """Fetch historical data from Bitget"""
        try:
            if not self.bitget_client:
                raise ValueError("Bitget client not provided in config")
            
            # Calculate number of candles to fetch based on timeframe
            # For 1h timeframe, we need 24 candles per day
            candles_per_day = 24 if self.timeframe == '1h' else 1
            
            # Calculate total days needed (analysis period + warmup)
            total_days_needed = self.days_to_analyze + self.warmup_period
            
            # Calculate the start date
            start_date = (datetime.now() - timedelta(days=total_days_needed)).strftime('%Y-%m-%d')
            
            logging.info(f"Fetching {self.timeframe} data from {start_date} for {self.symbol}")
            logging.info(f"Analysis period: {self.days_to_analyze} days with {self.warmup_period} days warmup")
            logging.info(f"Expected candles: ~{self.days_to_analyze * candles_per_day} for analysis + ~{self.warmup_period * candles_per_day} for warmup")
            
            # Get the client from config and access correct properties
            data = self.bitget_client.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_time=start_date
            )
            
            logging.info(f"Fetched {len(data)} candles for {self.symbol} with {self.timeframe} timeframe")
            
            # Verify we have enough data
            min_required_candles = (self.days_to_analyze + self.warmup_period) * candles_per_day * 0.9  # Allow for some missing candles (90%)
            if len(data) < min_required_candles:
                logging.warning(f"Fetched fewer candles ({len(data)}) than expected (min {min_required_candles}). Results may be incomplete.")
            
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_features(self, data):
        """Calculate required features for PCA analysis"""
        from utils.feature_calculator import calculate_all_features
        
        # Calculate all available features
        features_df = calculate_all_features(data.copy())
        
        # Calculate ema9_ratio (ema9 to closing price ratio)
        if 'ema9' in features_df.columns:
            # Get closing prices from the data
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Create ema9_ratio = ema9/close
            # Values above 1 indicate price is below EMA (bearish)
            # Values below 1 indicate price is above EMA (bullish)
            features_df['ema9_ratio'] = features_df['ema9'] / df['close'].values
            
            # Subtract 1 and multiply by 100 to get percentage difference from price
            features_df['ema9_ratio'] = (features_df['ema9_ratio'] - 1) * 100
            
            logging.info(f"Created ema9_ratio showing EMA9 vs price: min={features_df['ema9_ratio'].min():.2f}%, max={features_df['ema9_ratio'].max():.2f}%")
        
        # Add ema9_ratio to selected features if not there
        if 'ema9_ratio' in features_df.columns and 'ema9_ratio' not in self.selected_features:
            self.selected_features.append('ema9_ratio')
        
        # Remove original ema9 from selected features
        if 'ema9' in self.selected_features:
            self.selected_features.remove('ema9')
        
        # Select only the features we want for this analysis
        # Now we'll use all features for correlation analysis
        # selected_df = features_df[self.selected_features].copy()
        selected_df = features_df.copy()
        
        # Handle any NaN values - replace with 0
        selected_df = selected_df.fillna(0)
        
        # For 1h timeframe, adjust the warmup period based on candles
        warmup_candles = self.warmup_period
        if self.timeframe == '1h':
            warmup_candles = self.warmup_period * 24  # 24 hours per day
            logging.info(f"Adjusted warmup period for 1h timeframe: {warmup_candles} candles ({self.warmup_period} days)")
        
        # Skip warmup period to avoid initialization effects
        if len(selected_df) > warmup_candles:
            original_len = len(selected_df)
            selected_df = selected_df.iloc[warmup_candles:]
            logging.info(f"Skipped first {warmup_candles} rows as warmup period. Reduced dataset from {original_len} to {len(selected_df)} rows.")
        else:
            logging.warning(f"Dataset too small to skip warmup period. Consider reducing warmup_period (currently {self.warmup_period} days / {warmup_candles} candles).")
        
        # Detect and report any potential initialization values that might still exist
        potential_init_values = (selected_df == 0).sum() + (selected_df == -100).sum()
        if potential_init_values.sum() > 0:
            logging.warning("Potential initialization values found in data after warmup period:")
            for col, count in potential_init_values.items():
                if count > 0:
                    logging.warning(f"  - {col}: {count} potential initialization values")
        
        # For volume_ratio_50, apply a milder transformation to preserve more information
        # Use a square root transformation instead of log
        if 'volume_ratio_50' in selected_df.columns:
            # Apply square root transformation with sign preservation
            selected_df['volume_ratio_50'] = np.sign(selected_df['volume_ratio_50']) * np.sqrt(np.abs(selected_df['volume_ratio_50']))
            logging.info(f"Applied square root transformation to volume_ratio_50 to reduce extreme values while preserving more information")

        # Calculate additional transformations for the original data (needed for correlation analysis)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.iloc[warmup_candles:].reset_index(drop=True)
        selected_df = selected_df.reset_index(drop=True)

        # Transform supertrend to a ratio by dividing by the closing price
        if 'supertrend' in selected_df.columns:
            selected_df['supertrend'] = selected_df['supertrend'] / df['close'].values
            logging.info(f"Transformed supertrend to a ratio by dividing by closing price")

        # Normalize historical_volatility_30 by dividing by the mean of the closing price
        if 'historical_volatility_30' in selected_df.columns:
            mean_close = df['close'].mean()
            selected_df['historical_volatility_30'] = selected_df['historical_volatility_30'] / mean_close
            logging.info(f"Normalized historical_volatility_30 by dividing by mean closing price")

        # Transform vwap to a ratio by dividing by the closing price
        if 'vwap_ratio' in selected_df.columns:
            selected_df['vwap_ratio'] = selected_df['vwap_ratio']  # Already a ratio
            logging.info(f"vwap_ratio is already a ratio, no transformation needed")
        elif 'vwap' in selected_df.columns:
            selected_df['vwap'] = selected_df['vwap'] / df['close'].values
            logging.info(f"Transformed vwap to a ratio by dividing by closing price")
        
        # Filter out initial rows where ATR is zero
        if 'atr_ratio' in selected_df.columns:
            non_zero_atr_index = selected_df['atr_ratio'].ne(0).idxmax()
            selected_df = selected_df.iloc[non_zero_atr_index:]
            df = df.iloc[non_zero_atr_index:].reset_index(drop=True)
            selected_df = selected_df.reset_index(drop=True)
            logging.info(f"Filtered out initial rows where ATR is zero. Starting from index {non_zero_atr_index}.")
        
        # Apply log transformation to obv_ratio to scale it down
        if 'obv_ratio' in selected_df.columns:
            selected_df['obv_ratio'] = np.log1p(selected_df['obv_ratio'])
            logging.info("Applied log transformation to obv_ratio to scale it down.")
        
        # Fill any remaining NaN values with 0
        selected_df = selected_df.fillna(0)
        logging.info("Filled NaN values in selected_df with 0.")
        
        return selected_df, df

    def save_to_s3(self, df):
        """Save dataframe to S3 with timestamp-based folder structure"""
        try:
            # Create folder name based on current timestamp and timeframe
            now = datetime.now()
            folder_name = now.strftime("%y_%m_%d")
            file_name = f"PCA_{self.timeframe}_{now.strftime('%H_%M')}.csv"
            
            # Save DataFrame to a local CSV file first
            local_path = file_name
            df.to_csv(local_path)
            logging.info(f"Saved {self.timeframe} data to local file: {local_path}")
            
            # Full S3 path
            s3_path = f"{folder_name}/{file_name}"
            
            try:
                # Upload to S3
                logging.info(f"Uploading to S3: {s3_path}")
                self.s3_client.upload_file(
                    Filename=local_path,
                    Bucket=self.bucket_name,
                    Key=s3_path
                )
                
                # Verify upload
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_path)
                logging.info(f"Successfully verified upload to s3://{self.bucket_name}/{s3_path}")
                
                # Clean up local file
                os.remove(local_path)
                logging.info(f"Removed temporary local file: {local_path}")
                
                return f"s3://{self.bucket_name}/{s3_path}"
                
            except Exception as e:
                logging.error(f"Failed to upload to S3: {str(e)}")
                raise
            
        except Exception as e:
            logging.error(f"Error saving to S3: {str(e)}")
            raise


    def run_pca(self, data):
        """Run PCA analysis on the data using SageMaker remote execution"""
        # Call the standalone remote function instead of defining one here
        logging.info(f"Running PCA analysis on {self.timeframe} data in SageMaker...")
        
        # Log feature statistics before sending to SageMaker
        logging.info("Feature statistics before sending to SageMaker:")
        stats = data.describe().transpose()[['mean', 'std', 'min', 'max']]
        for feature, row in stats.iterrows():
            logging.info(f"{feature}: mean={row['mean']:.4f}, std={row['std']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}")
        
        result = run_pca_remote(
            data_input=data,
            role=self.role,
            n_components=self.n_components,
            bucket_name=self.bucket_name
        )
        logging.info(f"PCA result: {result}")
        return result


if __name__ == "__main__":
    # Load configuration from config.json
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']
    
    # Initialize BitgetFutures client
    bitget_client = BitgetFutures(api_setup)
    
    # Create config for SageMaker PCA - using 1h timeframe with 30 days of data
    config = {
        'symbol': 'ETH/USDT:USDT',
        'timeframe': '1h',
        'n_components': 5,
        'bitget_client': bitget_client,
        'warmup_period': 5,      # 5 days of warmup for hourly data
        'days_to_analyze': 30,   # 30 days of analysis data
        'correlation_threshold': 0.1,  # Correlation threshold for feature selection
        'future_return_periods': [1, 3, 5, 10]  # Periods for future returns
    }

    sagemaker_pca = SageMakerPCA(config)

    # 1. Fetch the data
    logging.info("\n===== FETCHING DATA =====")
    data = sagemaker_pca.fetch_data()
    logging.info(f"Fetched {len(data)} data points")
    
    # 2. Calculate features
    logging.info("\n===== CALCULATING FEATURES =====")
    features_df, price_df = sagemaker_pca.calculate_features(data)
    logging.info(f"Calculated features: {features_df.shape}")

    # 3. Perform correlation analysis and filter features
    logging.info("\n===== PERFORMING CORRELATION ANALYSIS =====")
    filtered_features_df = sagemaker_pca.perform_correlation_analysis(features_df, price_df)
    logging.info(f"Filtered features: {filtered_features_df.shape}")

    # Log feature statistics
    logging.info("\n===== FEATURE STATISTICS =====")
    stats = filtered_features_df.describe().transpose()[['mean', 'std', 'min', 'max']]
    for feature, row in stats.iterrows():
        logging.info(f"{feature}: mean={row['mean']:.4f}, std={row['std']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}")

    print(filtered_features_df.head())

    # 4. Save filtered features to CSV
    sagemaker_pca.save_to_s3(filtered_features_df)

    # 5. Run PCA on filtered features
    sagemaker_pca.run_pca(filtered_features_df)

