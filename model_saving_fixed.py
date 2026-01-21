import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'model_artifacts/recruitment_model.joblib'
ENCODERS_PATH = 'model_artifacts/label_encoders.joblib'

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def prepare_features(df):
    """
    Applies feature engineering to match training pipeline.
    Assumes df contains original raw columns.
    """
    df = df.copy()
    
    # Categorical encoding maps (from training)
    target_map = {'likely reject': 0, 'uncertain': 1, 'likely accept': 2}
    time_map = {'Fast': 0, 'Medium': 1, 'Slow': 2}
    cost_map = {'Low Cost': 0, 'Medium Cost': 1, 'High Cost': 2}
    
    # Apply mappings
    df['acceptance_category'] = df['acceptance_category'].str.lower().map(target_map)
    df['time_to_hire_category'] = df['time_to_hire_category'].map(time_map)
    df['cost_bucket'] = df['cost_bucket'].map(cost_map)
    
    # Standardize categorical columns
    cat_cols = ["job_title", "department", "source"]
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()
    
    return df

# ============================================================================
# LABEL ENCODING FUNCTION
# ============================================================================

def apply_label_encoding(df, encoders=None, fit=False):
    """
    Applies or fits label encoders for categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with categorical features
    encoders : dict or None
        Dictionary of fitted LabelEncoders. If None, new encoders will be created.
    fit : bool
        If True, fit new encoders. If False, use provided encoders.
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with encoded features
    encoders : dict
        Dictionary of LabelEncoders
    """
    df = df.copy()
    cat_cols = ["department", "source", "job_title"]
    
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        if encoders is None:
            raise ValueError("Encoders must be provided when fit=False")
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])
    
    return df, encoders

# ============================================================================
# MODEL SAVING FUNCTION
# ============================================================================

def save_model_artifacts(model, encoders, model_path=MODEL_PATH, encoders_path=ENCODERS_PATH):
    """
    Saves trained model and label encoders to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained XGBoost classifier
    encoders : dict
        Dictionary containing fitted LabelEncoders
    model_path : str
        Path to save model
    encoders_path : str
        Path to save encoders
    """
    import os
    
    # Create directory if not exists
    os.makedirs('model_artifacts', exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save encoders
    joblib.dump(encoders, encoders_path)
    print(f"✓ Label encoders saved to {encoders_path}")
    
    # Save feature names
    feature_names = [
        'department', 'job_title', 'num_applicants', 'source',
        'time_to_hire_category', 'cost_bucket', 'applicant_pressure_index',
        'cost_efficiency_daily', 'cost_per_applicant', 'hire_days_per_applicant',
        'difficulty_index_log', 'acceptance_cost_pressure', 'acceptance_time_pressure'
    ]
    
    metadata = {
        'feature_names': feature_names,
        'model_type': 'XGBClassifier',
        'target_classes': ['Likely Reject', 'Uncertain', 'Likely Accept']
    }
    
    joblib.dump(metadata, 'model_artifacts/metadata.joblib')
    print(f"✓ Metadata saved")

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_model_artifacts(model_path=MODEL_PATH, encoders_path=ENCODERS_PATH):
    """
    Loads trained model and encoders from disk.
    
    Returns:
    --------
    model : sklearn model
        Loaded XGBoost classifier
    encoders : dict
        Dictionary of LabelEncoders
    metadata : dict
        Model metadata
    """
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    metadata = joblib.load('model_artifacts/metadata.joblib')
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Encoders loaded from {encoders_path}")
    print(f"✓ Features: {len(metadata['feature_names'])}")
    
    return model, encoders, metadata

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_new_data(model, encoders, new_data):
    """
    Makes predictions on new data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    encoders : dict
        Fitted label encoders
    new_data : pd.DataFrame
        New data with raw features
    
    Returns:
    --------
    predictions : np.array
        Predicted classes
    probabilities : np.array
        Prediction probabilities
    """
    # Prepare features
    X_new = prepare_features(new_data)
    
    # Apply label encoding
    X_new, _ = apply_label_encoding(X_new, encoders=encoders, fit=False)
    
    # Select features (exclude target and redundant columns)
    feature_cols = [col for col in X_new.columns if col not in 
                   ['recruitment_id', 'offer_acceptance_rate', 'acceptance_category',
                    'time_to_hire_days', 'cost_per_hire', 'difficulty_index']]
    
    X_new = X_new[feature_cols]
    
    # Predict
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    return predictions, probabilities

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("MODEL SAVING DEMONSTRATION")
    print("="*70 + "\n")
    
    # Load your trained model and data
    # model = your_trained_xgboost_model
    # df_train = your_training_data
    
    # Example: Prepare and encode training data
    # df_processed = prepare_features(df_train)
    # X_encoded, label_encoders = apply_label_encoding(df_processed, fit=True)
    
    # Save model and encoders
    # save_model_artifacts(model, label_encoders)
    
    print("To save your model, uncomment and run:")
    print("  save_model_artifacts(final_model, label_encoders)")
    print("\nTo load model later:")
    print("  model, encoders, metadata = load_model_artifacts()")