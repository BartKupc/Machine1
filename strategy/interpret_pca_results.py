import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import io
import os
import tarfile
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add parent directory to path for imports
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))


# Set up constants
BUCKET_NAME = 'sagemaker-eu-west-1-688567281415'
# Direct path to the model.tar.gz file
MODEL_KEY = 'pca_output/pca-2025-03-14-09-21-21-722/output/model.tar.gz'
N_COMPONENTS = 5  # Match with what you used in training

def download_and_extract_model():
    """Download the model tar.gz file and extract it"""
    s3_client = boto3.client('s3')
    
    # Create a temporary directory for the model
    tmp_dir = os.path.join(os.getcwd(), 'tmp_model')
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Download the model tar.gz file
    tar_path = os.path.join(tmp_dir, 'model.tar.gz')
    logging.info(f"Downloading model from s3://{BUCKET_NAME}/{MODEL_KEY}")
    s3_client.download_file(BUCKET_NAME, MODEL_KEY, tar_path)
    
    # Extract the tar.gz file
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=tmp_dir)
    
    logging.info(f"Extracted model to {tmp_dir}")
    
    # Examine the contents of the extracted directory
    logging.info("Contents of the extracted model directory:")
    model_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            if file != 'model.tar.gz':  # Skip the original tar file
                file_path = os.path.join(root, file)
                logging.info(f"  - {file_path}")
                model_files.append(file_path)
    
    return tmp_dir, model_files

def fit_pca_with_sklearn():
    """Fit PCA on the data using scikit-learn"""
    logging.info("Using scikit-learn to fit PCA on the data")
    
    # Import scikit-learn
    from sklearn.decomposition import PCA
    import sklearn
    logging.info(f"Using scikit-learn {sklearn.__version__}")
    
    # Load the data
    data = load_latest_data()
    
    # Fit PCA
    logging.info(f"Fitting PCA with {N_COMPONENTS} components")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(data)
    
    # Extract components
    mean = pca.mean_
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    
    logging.info(f"Fitted PCA model with {N_COMPONENTS} components")
    logging.info(f"Mean shape: {mean.shape}")
    logging.info(f"Eigenvectors shape: {eigenvectors.shape}")
    logging.info(f"Eigenvalues shape: {eigenvalues.shape}")
    
    return mean, eigenvectors, eigenvalues, data

def load_latest_data():
    """Load the latest data file from S3 that was used for PCA"""
    s3_client = boto3.client('s3')
    
    # List all objects in the bucket
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME
    )
    
    # Filter for CSV files and find the most recent PCA_*.csv file
    csv_files = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('.csv') and 'PCA_' in key:
            csv_files.append((key, obj['LastModified']))
    
    if not csv_files:
        raise ValueError(f"No PCA data files found in s3://{BUCKET_NAME}")
    
    # Sort by last modified time to get the most recent
    latest_data_key = sorted(csv_files, key=lambda x: x[1], reverse=True)[0][0]
    logging.info(f"Found latest data file: {latest_data_key}")
    
    # Download the CSV file
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_data_key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    
    # First column is typically the index
    if 'timestamp' in df.columns or 'Unnamed: 0' in df.columns:
        index_col = 'timestamp' if 'timestamp' in df.columns else 'Unnamed: 0'
        df.set_index(index_col, inplace=True)
    
    logging.info(f"Loaded data with shape {df.shape}")
    return df

def visualize_explained_variance(eigenvalues):
    """Visualize the explained variance ratio"""
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Create the plot
    plt.figure(figsize=(12, 5))
    
    # Plot individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.title('Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.xticks(range(1, len(cumulative_variance) + 1))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('explained_variance.png')
    logging.info(f"Saved explained variance plot to explained_variance.png")
    plt.close()
    
    # Print explained variance ratios
    for i, var in enumerate(explained_variance_ratio):
        logging.info(f"PC{i+1} explains {var*100:.2f}% of variance")
    
    # Find number of components to explain 80% variance
    components_for_80_percent = np.argmax(cumulative_variance >= 0.8) + 1
    logging.info(f"Number of components to explain 80% variance: {components_for_80_percent}")
    
    return explained_variance_ratio, cumulative_variance

def visualize_component_loadings(eigenvectors, feature_names):
    """Visualize the component loadings (how much each feature contributes to each PC)"""
    num_components = min(N_COMPONENTS, len(eigenvectors))
    
    # Prepare a DataFrame for the loadings
    loadings_df = pd.DataFrame(
        eigenvectors[:num_components].T,
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=feature_names
    )
    
    # Create a heatmap of the loadings
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('PCA Component Loadings')
    plt.tight_layout()
    plt.savefig('component_loadings.png')
    logging.info(f"Saved component loadings plot to component_loadings.png")
    plt.close()
    
    # For each component, show the top contributing features
    for i in range(num_components):
        component = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
        logging.info(f"\nTop features for PC{i+1}:")
        for feature, loading in component.head(3).items():
            loading_actual = loadings_df.loc[feature, f'PC{i+1}']
            logging.info(f"{feature}: {loading_actual:.4f}")
    
    return loadings_df

def transform_data(data, mean, eigenvectors):
    """Transform the original data into PCA space"""
    # Center the data
    centered_data = data.values - mean
    
    # Project onto principal components
    transformed_data = np.dot(centered_data, eigenvectors.T)
    
    # Convert to DataFrame for easier handling
    transformed_df = pd.DataFrame(
        transformed_data,
        columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])],
        index=data.index
    )
    
    logging.info(f"Transformed data shape: {transformed_df.shape}")
    return transformed_df

def visualize_transformed_data(transformed_df):
    """Visualize the data in PCA space"""
    # Scatter plot of first two PCs
    plt.figure(figsize=(12, 10))
    
    # Main scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(transformed_df['PC1'], transformed_df['PC2'], alpha=0.6)
    plt.title('Data Projection onto PC1 and PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # PC1 vs PC3
    if 'PC3' in transformed_df.columns:
        plt.subplot(2, 2, 2)
        plt.scatter(transformed_df['PC1'], transformed_df['PC3'], alpha=0.6, color='green')
        plt.title('Data Projection onto PC1 and PC3')
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Distribution of PC1
    plt.subplot(2, 2, 3)
    sns.histplot(transformed_df['PC1'], kde=True)
    plt.title('Distribution of PC1')
    plt.xlabel('PC1')
    
    # Distribution of PC2
    plt.subplot(2, 2, 4)
    sns.histplot(transformed_df['PC2'], kde=True)
    plt.title('Distribution of PC2')
    plt.xlabel('PC2')
    
    plt.tight_layout()
    plt.savefig('pca_projections.png')
    logging.info(f"Saved PCA projections plot to pca_projections.png")
    plt.close()
    
    # For the first few components, check for outliers
    outliers = []
    for i in range(min(3, transformed_df.shape[1])):
        col = f'PC{i+1}'
        mean = transformed_df[col].mean()
        std = transformed_df[col].std()
        
        # Define outliers as points that are more than 3 std deviations away
        threshold = 3 * std
        col_outliers = transformed_df[(transformed_df[col] > mean + threshold) | 
                                      (transformed_df[col] < mean - threshold)]
        
        if not col_outliers.empty:
            logging.info(f"\nOutliers in {col}:")
            logging.info(col_outliers.index.tolist())
            outliers.extend(col_outliers.index.tolist())
    
    # Get unique outliers
    unique_outliers = list(set(outliers))
    if unique_outliers:
        logging.info(f"\nAll unique outliers: {unique_outliers}")
    
    return transformed_df

def generate_interpretation_report(explained_variance_ratio, loadings_df, outliers):
    """Generate a human-readable interpretation report"""
    # Create a file to save the interpretation
    with open('pca_interpretation.txt', 'w') as f:
        f.write("=== PCA INTERPRETATION REPORT ===\n\n")
        
        # Variance explained
        f.write("1. VARIANCE EXPLAINED\n")
        f.write("-----------------------\n")
        for i, var in enumerate(explained_variance_ratio):
            f.write(f"PC{i+1} explains {var*100:.2f}% of variance\n")
        f.write("\nInterpretation: ")
        
        # Interpret variance
        if explained_variance_ratio[0] > 0.9:
            f.write("The first component explains nearly all variance, suggesting a single dominant factor in the data.\n")
        elif explained_variance_ratio[0] + explained_variance_ratio[1] > 0.8:
            f.write("The first two components explain most of the variance, suggesting the data can be well-represented in 2D.\n")
        else:
            f.write("Multiple components are needed to explain the variance, indicating complex relationships in the data.\n")
        
        # Component interpretations
        f.write("\n2. PRINCIPAL COMPONENTS\n")
        f.write("-----------------------\n")
        
        # For each component with significant variance (>1%), interpret it
        for i in range(len(explained_variance_ratio)):
            if explained_variance_ratio[i] > 0.01:  # More than 1% variance
                f.write(f"\nPC{i+1} ({explained_variance_ratio[i]*100:.2f}% variance):\n")
                
                # Get top positive and negative contributors
                component = loadings_df[f'PC{i+1}']
                top_pos = component.sort_values(ascending=False).head(3)
                top_neg = component.sort_values(ascending=True).head(3)
                
                # Write positive contributors
                f.write("  Positive correlations:\n")
                for feature, value in top_pos.items():
                    if value > 0.1:  # Only if the loading is significant
                        f.write(f"  - {feature}: {value:.4f}\n")
                
                # Write negative contributors
                f.write("  Negative correlations:\n")
                for feature, value in top_neg.items():
                    if value < -0.1:  # Only if the loading is significant
                        f.write(f"  - {feature}: {value:.4f}\n")
                
                # Write interpretation
                f.write("  Interpretation: ")
                if i == 0 and explained_variance_ratio[0] > 0.9:
                    f.write(f"This component represents overall market data captured by {top_pos.index[0]}.\n")
                else:
                    pos_features = ', '.join([f for f, v in top_pos.items() if v > 0.3])
                    neg_features = ', '.join([f for f, v in top_neg.items() if v < -0.3])
                    
                    if pos_features and neg_features:
                        f.write(f"This component contrasts {pos_features} (positive) with {neg_features} (negative).\n")
                    elif pos_features:
                        f.write(f"This component primarily represents {pos_features}.\n")
                    elif neg_features:
                        f.write(f"This component primarily represents the inverse of {neg_features}.\n")
                    else:
                        f.write("This component has weak associations with all features.\n")
        
        # Outliers section
        f.write("\n3. OUTLIERS\n")
        f.write("-----------------------\n")
        if outliers:
            f.write(f"Found {len(outliers)} outlier dates:\n")
            # Convert outlier indices to strings before joining
            outlier_dates = [str(date) for date in outliers]
            f.write(', '.join(outlier_dates) + '\n')
            f.write("\nThese dates represent unusual market behavior that deviates significantly from typical patterns.\n")
            f.write("Potential events to investigate on these dates:\n")
            f.write("- Significant price movements or market events\n")
            f.write("- News or regulatory announcements\n")
            f.write("- Changes in trading volume or liquidity\n")
            f.write("- Technical market events (e.g., flash crashes)\n")
        else:
            f.write("No significant outliers detected in the data.\n")
        
        # Trading strategies section
        f.write("\n4. POTENTIAL TRADING STRATEGIES\n")
        f.write("-----------------------\n")
        f.write("Based on the PCA results, consider the following strategies:\n\n")
        
        # Based on first component
        top_pc1 = loadings_df['PC1'].abs().sort_values(ascending=False).index[0]
        f.write(f"1. {top_pc1}-based strategy:\n")
        f.write(f"   Track changes in {top_pc1} as it's the most significant factor in the data.\n")
        
        # Based on outliers
        f.write("\n2. Anomaly detection strategy:\n")
        f.write("   Monitor real-time data for patterns similar to the identified outliers,\n")
        f.write("   which may signal unusual market conditions that present opportunities.\n")
        
        # Based on PC2 if significant
        if len(explained_variance_ratio) > 1 and explained_variance_ratio[1] > 0.05:
            top_pc2 = loadings_df['PC2'].abs().sort_values(ascending=False).index[0]
            f.write(f"\n3. Secondary factor strategy:\n")
            f.write(f"   Track {top_pc2} as a secondary indicator to refine entry/exit timing.\n")
        
        f.write("\n5. NEXT STEPS\n")
        f.write("-----------------------\n")
        f.write("1. Validate findings with backtest on historical data\n")
        f.write("2. Consider developing trading signals based on transformed PC values\n")
        f.write("3. Monitor outlier dates to see if patterns recur\n")
        f.write("4. Consider if data preprocessing or feature scaling might improve results\n")
        
    logging.info(f"Saved interpretation report to pca_interpretation.txt")
    return

def main():
    """Main function to interpret PCA results"""
    try:
        # Download and extract the model directly from the known path
        model_dir, model_files = download_and_extract_model()
        
        # Fit PCA with scikit-learn
        mean, eigenvectors, eigenvalues, data = fit_pca_with_sklearn()
        feature_names = data.columns.tolist()
        
        # Visualize explained variance
        explained_variance_ratio, cumulative_variance = visualize_explained_variance(eigenvalues)
        
        # Visualize component loadings
        loadings_df = visualize_component_loadings(eigenvectors, feature_names)
        
        # Transform data to PCA space
        transformed_df = transform_data(data, mean, eigenvectors)
        
        # Visualize transformed data
        visualize_transformed_data(transformed_df)
        
        # Find outliers
        outliers = []
        for i in range(min(3, transformed_df.shape[1])):
            col = f'PC{i+1}'
            mean_val = transformed_df[col].mean()
            std_val = transformed_df[col].std()
            
            # Define outliers as points that are more than 3 std deviations away
            threshold = 3 * std_val
            col_outliers = transformed_df[(transformed_df[col] > mean_val + threshold) | 
                                        (transformed_df[col] < mean_val - threshold)]
            
            outliers.extend(col_outliers.index.tolist())
        
        # Get unique outliers
        unique_outliers = list(set(outliers))
        
        # Generate interpretation report
        generate_interpretation_report(explained_variance_ratio, loadings_df, outliers)
        
        # Save transformed data to CSV
        transformed_df.to_csv('pca_transformed_data.csv')
        logging.info(f"Saved transformed data to pca_transformed_data.csv")
        
        # Clean up
        import shutil
        shutil.rmtree(model_dir)
        logging.info(f"Cleaned up temporary model directory")
        
        logging.info("\nPCA Interpretation Summary:")
        logging.info(f"- Number of components: {N_COMPONENTS}")
        logging.info(f"- Top 2 components explain {cumulative_variance[1]*100:.2f}% of variance")
        logging.info(f"- See plots and pca_interpretation.txt for detailed insights")
        
    except Exception as e:
        logging.error(f"Error interpreting PCA results: {str(e)}")
        raise

if __name__ == "__main__":
    main()