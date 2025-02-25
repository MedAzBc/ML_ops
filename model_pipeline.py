import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

def data_overview(df):
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}\n")
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")
    print(f"Duplicated rows count: {df.duplicated().sum()}")

def preprocess_data(df, scaler=None, pca_components=None):
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables (e.g., 'International plan' and 'Voice mail plan')
    df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
    df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})
    
    # Detect and handle outliers (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Feature selection (excluding non-numeric columns and target variable)
    feature_cols = [col for col in df.columns if col not in ['State', 'Churn']]
    X = df[feature_cols]
    y = df['Churn'].astype(int)
    
    # Normalize data
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Apply PCA if requested
    if pca_components:
        pca = PCA(n_components=pca_components)
        X_transformed = pca.fit_transform(X_scaled)
    else:
        pca = None
        X_transformed = X_scaled
    
    return pd.DataFrame(X_transformed), y, scaler, pca

def prepare_data(train_path, test_path, pca_components=None):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    data_overview(df_train)
    data_overview(df_test)
    
    X_train, y_train, scaler, pca = preprocess_data(df_train, pca_components=pca_components)
    X_test, y_test, _, _ = preprocess_data(df_test, scaler=scaler, pca_components=pca_components)
    
    return X_train, y_train, X_test, y_test, scaler, pca

def save_model(model, scaler, pca, path="model.joblib"):
    joblib.dump((model, scaler, pca), path)

def load_model(path="model.joblib"):
    return joblib.load(path)

