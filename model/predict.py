#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import joblib
import argparse
import logging
import tempfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_selection import (
    process_dataset,
    normalize_features,
    select_features_with_fratio,
    detect_outliers,
    extract_shift_invariant_features,
    extract_duration_invariant_features,
    extract_environmental_features,
    extract_absolute_env_features
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'prediction.log'))
    ]
)
logger = logging.getLogger('peanut_detector_prediction')

class TransformationParams:
    def __init__(self):
        self.feature_indices = None
        self.model_params = None
        self.model_type = None
        self.scaler = None
        self.max_features = None
        self.random_fraction = None
        self.include_env = None
        self.absolute_t = None
        self.feature_description = None
        self.decision_threshold = 0.5
        self.is_pipeline = False
        self.pipeline_steps = []

    def load_from_config(self, feature_config):
        if feature_config is not None:
            self.feature_indices = feature_config.get('indices')
            self.model_type = feature_config.get('model_type')
            self.model_params = feature_config.get('model_params')
            self.features_shape = feature_config.get('features_shape')
            self.classes = feature_config.get('classes')
            
            self.is_pipeline = "Pipeline" in self.model_type if self.model_type else False
            
            if self.is_pipeline:
                if "PolynomialFeatures" in self.model_type:
                    self.pipeline_steps = ["poly", "model"]
                else:
                    self.pipeline_steps = []
            else:
                self.pipeline_steps = []
            
            self.max_features = feature_config.get('max_features', 50)
            self.random_fraction = feature_config.get('random_fraction', 0.2)
            self.include_env = feature_config.get('include_env', True)
            self.absolute_t = feature_config.get('absolute_t', True)
                
            if 'feature_description' in feature_config:
                self.feature_description = feature_config.get('feature_description')
                
            logger.info(f"Loaded transformation parameters from config")
            
            if self.is_pipeline:
                logger.info(f"  Model type: {self.model_type}")
                logger.info(f"  Pipeline steps: {self.pipeline_steps}")
            else:
                logger.info(f"  Model type: {self.model_type}")
                
            logger.info(f"  Selected features: {len(self.feature_indices)}")
            logger.info(f"  Max features: {self.max_features}")
            logger.info(f"  Random fraction: {self.random_fraction}")
            logger.info(f"  Include env: {self.include_env}")
            logger.info(f"  Absolute T: {self.absolute_t}")
            logger.info(f"  Features shape: {self.features_shape}")
            logger.info(f"  Classes: {self.classes}")
            
        else:
            logger.error("Invalid feature configuration - cannot load transformation parameters")
            raise ValueError("Invalid feature configuration")

    def load_scaler(self, scaler_path=None):
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_scaler.joblib')
            
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"Error loading scaler: {str(e)}. This will cause inconsistent predictions.")
                raise RuntimeError(f"Failed to load scaler: {str(e)}")
        else:
            logger.error(f"No saved scaler found at {scaler_path}. This will cause inconsistent predictions.")
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Re-train the model first.")
            
    def load_decision_threshold(self, threshold_path=None):
        if threshold_path is None:
            threshold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_threshold.joblib')
            
        if os.path.exists(threshold_path):
            try:
                self.decision_threshold = joblib.load(threshold_path)
                logger.info(f"Loaded decision threshold: {self.decision_threshold}")
            except Exception as e:
                logger.warning(f"Error loading threshold: {str(e)}. Using default threshold 0.5.")
                self.decision_threshold = 0.5
        else:
            logger.warning(f"No saved threshold found at {threshold_path}. Using default threshold 0.5.")
            self.decision_threshold = 0.5

def load_model_and_config(model_path=None, config_path=None, scaler_path=None, threshold_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'peanut_detector.joblib')
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_config.joblib')
        
    if scaler_path is None:
        scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_scaler.joblib')
        
    if threshold_path is None:
        threshold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_threshold.joblib')
    
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            steps = [step[0] for step in model.steps]
            logger.info(f"Loaded Pipeline model with steps: {steps}")
            if 'poly' in steps:
                logger.info("Pipeline includes polynomial feature transformation")
        else:
            logger.info(f"Loaded direct model: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    try:
        logger.info(f"Loading feature configuration from {config_path}")
        feature_config = joblib.load(config_path)
        logger.info(f"Loaded feature configuration with {len(feature_config['indices'])} selected features")
    except Exception as e:
        logger.error(f"Error loading feature configuration: {str(e)}")
        raise RuntimeError(f"Failed to load feature configuration: {str(e)}")
    
    transform_params = TransformationParams()
    transform_params.load_from_config(feature_config)
    transform_params.load_scaler(scaler_path)
    transform_params.load_decision_threshold(threshold_path)
    
    if not verify_model_consistency(model, transform_params):
        logger.warning("Model and transformation parameters are inconsistent. Predictions may be unreliable.")
    
    return model, transform_params

def verify_model_consistency(model, transform_params):
    from sklearn.pipeline import Pipeline
    
    if isinstance(model, Pipeline):
        logger.info("Detected Pipeline model with multiple steps")
        final_estimator = model.steps[-1][1]
        
        model_type_matches = transform_params.model_type == type(final_estimator).__name__
        if not model_type_matches:
            logger.warning(f"Model type mismatch: config has {transform_params.model_type}, but loaded final estimator is {type(final_estimator).__name__}")
            return False
        
        if hasattr(final_estimator, 'coef_'):
            if transform_params.classes is not None and hasattr(final_estimator, 'classes_'):
                if not np.array_equal(final_estimator.classes_, np.array(transform_params.classes)):
                    logger.warning(f"Class mismatch: config has {transform_params.classes}, but model has {final_estimator.classes_}")
                    return False
            
            return True
    else:
        if transform_params.model_type != type(model).__name__:
            logger.warning(f"Model type mismatch: config has {transform_params.model_type}, but loaded {type(model).__name__}")
            return False
        
        if hasattr(model, 'coef_'):
            expected_features = len(transform_params.feature_indices)
            actual_features = model.coef_.shape[1] if model.coef_.ndim > 1 else model.coef_.shape[0]
            
            if expected_features != actual_features:
                logger.warning(f"Feature count mismatch: config has {expected_features}, but model expects {actual_features}")
                return False
            
            if transform_params.classes is not None and hasattr(model, 'classes_'):
                if not np.array_equal(model.classes_, np.array(transform_params.classes)):
                    logger.warning(f"Class mismatch: config has {transform_params.classes}, but model has {model.classes_}")
                    return False
    
    return True

def load_and_preprocess_data(input_path, transform_params):
    logger.info(f"Loading data from {input_path}")
    
    try:
        if os.path.exists(input_path):
            data = pd.read_csv(input_path)
            logger.info(f"Loaded CSV with shape: {data.shape}")
        else:
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        filename = os.path.basename(input_path)
        
        if '_' in filename:
            substance_name = filename.split('_')[0]
        else:
            substance_name = os.path.splitext(filename)[0]
        
        base_substance_map = {
            "Peanut Butter 0.97": "Peanut Butter",
            "Peanut Butter 0.38": "Peanut Butter",
            "Peanut Butter 1.43": "Peanut Butter",
            "Air (Negative Control)": "Air",
            "Peanut Sauce": "Peanut Sauce",
            "Peanut Oil": "Peanut Oil",
            "Yum Yum Sauce": "Yum Yum Sauce",
            "Mineral Oil": "Mineral Oil"
        }
        
        base_substance = base_substance_map.get(substance_name, substance_name)
        
        logger.info(f"Detected substance: {substance_name} (base: {base_substance})")
        
        categories = {
            "Bread": 0,
            "Air": 0,
            "Yum Yum Sauce": 0,
            "Peanut Butter": 1,
            "Peanut Sauce": 1,
            "Peanut Oil": 1,
            "Mineral Oil": 0
        }
        
        if base_substance not in categories:
            logger.warning(f"Unknown substance '{base_substance}'. Defaulting to 'unknown' with value 0.")
            categories[base_substance] = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            substance_dir = os.path.join(temp_dir, base_substance)
            os.makedirs(substance_dir, exist_ok=True)
            
            temp_csv_path = os.path.join(substance_dir, filename)
            data.to_csv(temp_csv_path, index=False)
            logger.info(f"Prepared CSV for processing at {temp_csv_path}")
            
            features, category_labels, _, _ = process_dataset(temp_dir, categories)
            logger.info(f"Extracted raw features with shape: {features.shape}")
            
            if features.shape[0] == 0:
                logger.error("No features extracted from input data")
                raise ValueError("No features extracted from input data")
            
            if transform_params.scaler is not None:
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                features_scaled = transform_params.scaler.transform(features)
                logger.info(f"Applied trained scaler to features with shape: {features_scaled.shape}")
            else:
                logger.error("Scaler is not available. Cannot normalize features.")
                raise ValueError("Scaler is not available. Rerun training to save the scaler.")
            
            feature_indices = transform_params.feature_indices
            
            if feature_indices is not None and len(feature_indices) > 0:
                if max(feature_indices) < features_scaled.shape[1]:
                    selected_features = features_scaled[:, feature_indices]
                    logger.info(f"Selected {len(feature_indices)} features as used in training")
                else:
                    logger.error(f"Feature indices out of range. Max index {max(feature_indices)} > {features_scaled.shape[1]}")
                    raise IndexError(f"Feature indices out of range. Rerun training to update feature selection.")
            else:
                logger.error("No feature indices available for selection")
                raise ValueError("No feature indices available. Rerun training to generate feature indices.")
                
            return selected_features
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to preprocess data: {str(e)}")

def make_prediction(model, features, transform_params):
    logger.info(f"Making prediction with feature matrix of shape {features.shape}")
    
    try:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        probabilities = model.predict_proba(features)
        positive_prob = probabilities[0][1]
        
        threshold = transform_params.decision_threshold
        
        prediction = 1 if positive_prob >= threshold else 0
        
        if prediction == 1:
            prediction_label = "Contains peanut"
        else:
            prediction_label = "No peanut detected"
        
        result = {
            'prediction': prediction,
            'prediction_label': prediction_label,
            'probability': positive_prob,
            'threshold': threshold
        }
        
        logger.info(f"Prediction: {prediction_label} (probability: {positive_prob:.4f}, threshold: {threshold:.4f})")
        
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise RuntimeError(f"Failed to make prediction: {str(e)}")

def save_predictions(input_path, output_path, result, consolidated_output=None):
    filename = os.path.basename(input_path)
    
    if '_' in filename:
        substance_name = filename.split('_')[0]
    else:
        substance_name = os.path.splitext(filename)[0]
    
    base_substance_map = {
        "Peanut Butter 0.97": "Peanut Butter",
        "Peanut Butter 0.38": "Peanut Butter",
        "Peanut Butter 1.43": "Peanut Butter",
        "Air (Negative Control)": "Air",
        "Peanut Sauce": "Peanut Sauce",
        "Peanut Oil": "Peanut Oil",
        "Yum Yum Sauce": "Yum Yum Sauce",
        "Mineral Oil": "Mineral Oil"
    }
    
    prediction_entry = {
        'filename': filename,
        'substance': substance_name,
        'sample_index': 0,
        'prediction': result['prediction'],
        'probability': result['probability'],
        'predicted_class': 'peanut' if result['prediction'] == 1 else 'non-peanut'
    }
    
    if output_path is None:
        parent_dir = os.path.dirname(input_path)
        output_dir = os.path.join(parent_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_results.csv")
    
    results_df = pd.DataFrame([prediction_entry])
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction results to {output_path}")
    
    if consolidated_output is not None:
        if os.path.exists(consolidated_output):
            consolidated_df = pd.read_csv(consolidated_output)
            
            existing_rows = consolidated_df[consolidated_df['filename'] == filename]
            if len(existing_rows) > 0:
                consolidated_df.loc[consolidated_df['filename'] == filename, prediction_entry.keys()] = prediction_entry.values()
                logger.info(f"Updated existing entry in consolidated results for {filename}")
            else:
                consolidated_df = pd.concat([consolidated_df, pd.DataFrame([prediction_entry])], ignore_index=True)
                logger.info(f"Added new entry to consolidated results for {filename}")
        else:
            consolidated_df = pd.DataFrame([prediction_entry])
            logger.info(f"Created new consolidated results file with entry for {filename}")
            
        consolidated_df.to_csv(consolidated_output, index=False)
        logger.info(f"Saved consolidated results to {consolidated_output}")
    
    return {
        'individual_results': output_path,
        'consolidated_results': consolidated_output
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict peanut presence in sensor data')
    
    parser.add_argument('--input', '-i', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to save prediction results')
    parser.add_argument('--model', '-m', help='Path to the saved model file')
    parser.add_argument('--config', '-c', help='Path to the saved feature configuration file')
    parser.add_argument('--scaler', '-s', help='Path to the saved scaler file')
    parser.add_argument('--threshold', '-t', help='Path to the saved decision threshold file')
    parser.add_argument('--consolidated', help='Path to save consolidated results across multiple predictions')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        model, transform_params = load_model_and_config(
            model_path=args.model,
            config_path=args.config,
            scaler_path=args.scaler,
            threshold_path=args.threshold
        )
        
        features = load_and_preprocess_data(args.input, transform_params)
        
        result = make_prediction(model, features, transform_params)
        
        save_predictions(args.input, args.output, result, args.consolidated)
        
        logger.info(f"Prediction completed successfully: {result['prediction_label']} ({result['probability']:.4f})")
        
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        
        return result
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()