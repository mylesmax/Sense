#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import logging
from sklearn.model_selection import ParameterGrid
import glob

VALIDATION_SIZE = 0.1

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_selection import (
    process_dataset,
    normalize_features,
    select_features_with_fratio,
    detect_outliers,
    load_and_process_data
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'training.log'))
    ]
)
logger = logging.getLogger('peanut_detector')

def load_translational_data(translational_path=None):
    if translational_path is None:
        translational_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'translationalData')
    
    if not os.path.exists(translational_path):
        logger.warning(f"Translational data directory {translational_path} not found, skipping...")
        return [], [], []
    
    logger.info(f"Loading translational data from {translational_path}")
    
    features_list = []
    binary_labels = []
    sample_paths = []
    
    files = glob.glob(os.path.join(translational_path, "*.csv"))
    yes_count = 0
    no_count = 0
    
    for file in files:
        filename = os.path.basename(file)
        
        if filename.startswith("YES_"):
            label = 1
            yes_count += 1
        elif filename.startswith("NO_"):
            label = 0
            no_count += 1
        else:
            logger.warning(f"Skipping file {filename} - doesn't have YES_ or NO_ prefix")
            continue
        
        feature_vector = load_and_process_data(file)
        
        if feature_vector is not None:
            features_list.append(feature_vector)
            binary_labels.append(label)
            sample_paths.append(file)
    
    logger.info(f"Processed {len(features_list)} translational samples: {yes_count} positive, {no_count} negative")
    
    return features_list, binary_labels, sample_paths

def load_data(data_path=None, max_features=100, random_fraction=0.8, include_env=True, absolute_t=True, 
              translational_path=None, translational_weight=1.0):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    
    logger.info(f"Loading data from {data_path}")
    
    categories = {
        "Bread": 0,
        "Air": 0,
        "Yum Yum Sauce": 0,
        "Peanut Butter": 1,
        "Peanut Sauce": 1,
        "Peanut Oil": 1,
        "Mineral Oil": 0
    }
    
    features, category_labels, binary_labels, feature_descriptions = process_dataset(data_path, categories)
    
    trans_features_list, trans_binary_labels, trans_sample_paths = load_translational_data(translational_path)
    
    sample_weights = np.ones(len(binary_labels) + len(trans_binary_labels))
    
    if trans_features_list:
        logger.info(f"Combining standard data ({len(binary_labels)} samples) with translational data ({len(trans_binary_labels)} samples)")
        
        sample_weights[len(binary_labels):] = translational_weight
        logger.info(f"Assigned weight {translational_weight} to translational samples")
        
        if features.shape[0] > 0 and len(trans_features_list) > 0:
            features = np.vstack((features, np.vstack(trans_features_list)))
            binary_labels = binary_labels + trans_binary_labels
        elif len(trans_features_list) > 0:
            features = np.vstack(trans_features_list)
            binary_labels = trans_binary_labels
            
    y = np.array(binary_labels)
    
    logger.info(f"Feature matrix shape: {features.shape}")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    logger.info(f"Fitted feature scaler with {features.shape[1]} features")
    
    feature_desc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_descriptions.joblib')
    joblib.dump(feature_descriptions, feature_desc_path)
    logger.info(f"Saved feature descriptions to {feature_desc_path}")
    
    selected_features, selected_indices, fratios = select_features_with_fratio(
        features_scaled, category_labels + ['Translational'] * len(trans_binary_labels), 
        max_features=max_features, 
        random_fraction=random_fraction, include_env=include_env, 
        use_absolute_t_75=absolute_t
    )
    
    if selected_features.shape[0] > 0:
        try:
            corr_matrix = np.corrcoef(selected_features, rowvar=False)
            corr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_correlation.joblib')
            joblib.dump(corr_matrix, corr_path)
            logger.info(f"Saved feature correlation matrix to {corr_path}")
        except Exception as e:
            logger.warning(f"Could not compute feature correlation: {str(e)}")
    
    inlier_mask = detect_outliers(selected_features, category_labels + ['Translational'] * len(trans_binary_labels))
    X = selected_features[inlier_mask]
    y = y[inlier_mask]
    sample_weights = sample_weights[inlier_mask]
    
    logger.info(f"Removed {sum(~inlier_mask)} outliers out of {len(inlier_mask)} samples")
    logger.info(f"Final feature matrix shape: {X.shape}")
    
    return X, y, sample_weights, selected_indices, fratios, scaler

def train_model(X, y, sample_weights=None, model_params=None, cv=5, validation_size=None):
    if validation_size is None:
        validation_size = VALIDATION_SIZE
    
    if sample_weights is not None:
        X_train, X_val, y_train, y_val, sample_weights_train, sample_weights_val = train_test_split(
            X, y, sample_weights, test_size=validation_size, random_state=42, stratify=y
        )
        logger.info(f"Using sample weights: min={sample_weights_train.min()}, max={sample_weights_train.max()}, mean={sample_weights_train.mean()}")
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=42, stratify=y)
        sample_weights_train = None
    
    train_percentage = int((1 - validation_size) * 100)
    val_percentage = int(validation_size * 100)
    logger.info(f"Split ratio: {train_percentage}/{val_percentage} (train/validation)")
    logger.info(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    logger.info(f"Class distribution - Training: {np.bincount(y_train)}, Validation: {np.bincount(y_val)}")
    
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('model', LogisticRegression(random_state=42, max_iter=5000))
    ])
    
    if model_params is None:
        model_params = {
            'poly__degree': [2],
            'poly__interaction_only': [True],
            'model__C': [1, 5, 10, 20],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear'],
            'model__class_weight': ['balanced', {0: 1, 1: 2}]
        }
    
    logger.info("Performing grid search for polynomial logistic regression")
    logger.info("Note: Feature matrix will expand with polynomial terms")
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    expanded_shape = poly.fit_transform(X_train[:1]).shape[1]
    logger.info(f"Original features: {X_train.shape[1]}, Expanded polynomial features: {expanded_shape}")
    
    grid_search = GridSearchCV(
        pipeline, 
        model_params, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=0
    )
    
    if sample_weights_train is not None:
        grid_search.fit(X_train, y_train, model__sample_weight=sample_weights_train)
        logger.info("Trained model with custom sample weights")
    else:
        grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Cross-validation score: {grid_search.best_score_:.4f}")
    
    poly_features = model.named_steps['poly']
    n_poly_features = poly_features.transform(X_train[:1]).shape[1]
    logger.info(f"Expanded to {n_poly_features} polynomial features")
    
    if sample_weights_train is not None:
        return model, X_train, X_val, y_train, y_val, sample_weights_train
    else:
        return model, X_train, X_val, y_train, y_val, None

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    
    logger.info(f"Validation Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC AUC:   {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:")
    logger.info(f"{conf_matrix}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return metrics

def plot_results(model, X_train, X_val, y_train, y_val, metrics, feature_indices, fratios):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(metrics['fpr'], metrics['tpr'], 
                   label=f"ROC curve (AUC = {metrics['roc_auc']:.2f})")
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Receiver Operating Characteristic')
    axes[0, 0].legend(loc="lower right")
    
    conf_matrix = metrics['confusion_matrix']
    im = axes[0, 1].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 1].set_title('Confusion Matrix')
    
    tick_marks = np.arange(2)
    axes[0, 1].set_xticks(tick_marks)
    axes[0, 1].set_yticks(tick_marks)
    axes[0, 1].set_xticklabels(['Non-Peanut', 'Peanut'])
    axes[0, 1].set_yticklabels(['Non-Peanut', 'Peanut'])
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axes[0, 1].text(j, i, format(conf_matrix[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if conf_matrix[i, j] > thresh else "black")
    
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_ylabel('True label')
    axes[0, 1].set_xlabel('Predicted label')
    
    if isinstance(model, Pipeline) and hasattr(model.named_steps['model'], 'coef_'):
        lr_model = model.named_steps['model']
        coef = lr_model.coef_[0]
        
        sorted_coef = sorted(zip(range(len(coef)), coef), key=lambda x: abs(x[1]), reverse=True)
        
        top_n = min(20, len(sorted_coef))
        top_indices = [f[0] for f in sorted_coef[:top_n]]
        top_coefs = [f[1] for f in sorted_coef[:top_n]]
        
        logger.info("Top polynomial feature importances (coefficients):")
        for idx, (feature_idx, coef_val) in enumerate(sorted_coef[:10]):
            logger.info(f"  Polynomial feature {feature_idx}: Coefficient = {coef_val:.4f}")
        
        y_pos = np.arange(top_n)
        axes[1, 0].barh(y_pos, top_coefs, align='center')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([f"Poly. Feature {idx}" for idx in top_indices])
        axes[1, 0].set_xlabel('Coefficient')
        axes[1, 0].set_title('Feature Importance (Model Coefficients)')
        
        top_n_original = min(20, len(feature_indices))
        y_pos = np.arange(top_n_original)
        top_fratios = sorted(zip(feature_indices, fratios), key=lambda x: x[1], reverse=True)[:top_n_original]
        axes[1, 1].barh(y_pos, [f[1] for f in top_fratios], align='center')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"Feature {idx}" for idx, _ in top_fratios])
        axes[1, 1].set_xlabel('F-ratio')
        axes[1, 1].set_title('Original Feature Importance (F-ratio)')
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=300)
    logger.info(f"Saved model evaluation plots to {os.path.join(output_dir, 'model_evaluation.png')}")
    
    plt.close(fig)

def save_model(model, feature_indices, scaler, output_dir=None, max_features=100, random_fraction=0.8, include_env=True, absolute_t=True):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'peanut_detector.joblib')
    joblib.dump(model, model_path)
    
    config_path = os.path.join(output_dir, 'feature_config.joblib')
    
    if isinstance(model, Pipeline):
        lr_model = model.named_steps['model']
        poly_model = model.named_steps['poly']
        model_params = {
            'poly_degree': poly_model.degree,
            'poly_interaction_only': poly_model.interaction_only,
            'poly_include_bias': poly_model.include_bias
        }
        
        for param_name in ['C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling',
                         'l1_ratio', 'max_iter', 'multi_class', 'n_jobs', 'penalty',
                         'random_state', 'solver', 'tol', 'verbose', 'warm_start']:
            if hasattr(lr_model, param_name):
                model_params[param_name] = getattr(lr_model, param_name)
    else:
        model_params = {}
        for param_name in ['C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling',
                         'l1_ratio', 'max_iter', 'multi_class', 'n_jobs', 'penalty',
                         'random_state', 'solver', 'tol', 'verbose', 'warm_start']:
            if hasattr(model, param_name):
                model_params[param_name] = getattr(model, param_name)
    
    if isinstance(model, Pipeline) and hasattr(model.named_steps['model'], 'coef_'):
        features_shape = model.named_steps['model'].coef_.shape[1]
        classes = model.named_steps['model'].classes_.tolist() if hasattr(model.named_steps['model'], 'classes_') else None
    elif hasattr(model, 'coef_'):
        features_shape = model.coef_.shape[1]
        classes = model.classes_.tolist() if hasattr(model, 'classes_') else None
    else:
        features_shape = None
        classes = None
        
    feature_config = {
        'indices': feature_indices,
        'model_type': 'Pipeline with PolynomialFeatures and LogisticRegression' if isinstance(model, Pipeline) else type(model).__name__,
        'model_params': model_params,
        'max_features': max_features,
        'random_fraction': random_fraction,
        'include_env': include_env,
        'absolute_t': absolute_t,
        'features_shape': features_shape,
        'classes': classes
    }
    
    joblib.dump(feature_config, config_path)
    
    scaler_path = os.path.join(output_dir, 'feature_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")
    
    threshold_path = os.path.join(output_dir, 'decision_threshold.joblib')
    threshold = 0.5
    joblib.dump(threshold, threshold_path)
    logger.info(f"Saved decision threshold to {threshold_path}")
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Feature configuration saved to {config_path}")
    
    return model_path, config_path

def main(data_path=None, max_features=100, random_fraction=0.8, include_env=True, absolute_t=True, 
         translational_path=None, translational_weight=1.0, validation_size=None):
    logger.info("Starting peanut detection model training")
    
    val_size = validation_size if validation_size is not None else VALIDATION_SIZE
    train_percentage = int((1 - val_size) * 100)
    val_percentage = int(val_size * 100)
    logger.info(f"Using {train_percentage}/{val_percentage} train/validation split")
    
    X, y, sample_weights, feature_indices, fratios, scaler = load_data(
        data_path=data_path,
        max_features=max_features,
        random_fraction=random_fraction,
        include_env=include_env,
        absolute_t=absolute_t,
        translational_path=translational_path,
        translational_weight=translational_weight
    )
    
    model, X_train, X_val, y_train, y_val, sample_weights_train = train_model(
        X, y, sample_weights, validation_size=validation_size
    )
    
    metrics = evaluate_model(model, X_val, y_val)
    
    plot_results(model, X_train, X_val, y_train, y_val, metrics, feature_indices, fratios)
    
    save_model(
        model, 
        feature_indices, 
        scaler,
        max_features=max_features,
        random_fraction=random_fraction,
        include_env=include_env,
        absolute_t=absolute_t
    )
    
    logger.info("Model training and evaluation completed successfully")
    
    return model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a peanut detection model')
    parser.add_argument('--data-path', type=str, default=None, 
                        help='Path to the data directory')
    parser.add_argument('--max-features', type=int, default=100, 
                        help='Maximum number of features to select')
    parser.add_argument('--random-fraction', type=float, default=0.8, 
                        help='Fraction of data used to generate the F-ratio ranking')
    parser.add_argument('--include-env', action='store_true', default=True, 
                        help='Include environmental features')
    parser.add_argument('--no-env', action='store_false', dest='include_env',
                        help='Exclude environmental features')
    parser.add_argument('--absolute-t', action='store_true', default=True, 
                        help='Use absolute temperature features')
    parser.add_argument('--no-absolute-t', action='store_false', dest='absolute_t',
                        help='Do not use absolute temperature features')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    parser.add_argument('--translational-path', type=str, default=None,
                        help='Path to the translational data directory')
    parser.add_argument('--translational-weight', type=float, default=1.0,
                        help='Weight multiplier for translational samples (default: 1.0)')
    parser.add_argument('--validation-size', type=float, default=None,
                        help='Size of validation set, e.g., 0.2 for 80/20 split (default: 0.2)')
    
    args = parser.parse_args()
    
    logger.setLevel(getattr(logging, args.log_level))
    
    main(
        data_path=args.data_path,
        max_features=args.max_features, 
        random_fraction=args.random_fraction,
        include_env=args.include_env,
        absolute_t=args.absolute_t,
        translational_path=args.translational_path,
        translational_weight=args.translational_weight,
        validation_size=args.validation_size
    )