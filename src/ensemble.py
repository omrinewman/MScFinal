# ensembling.py


# Data Handling & Numerical Operations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Model Training
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Metrics & Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)

# Distance Calculations
from scipy.spatial import distance

# Explainability
import shap


class Ensembling:
    def __init__(self, state):
        self.scaler = StandardScaler()
        self.random_state = state
        self.val_size = 0.15 # Validation set proportion
        self.test_size = 0.10 # Test set proportion
        self.n_jobs = 1 # Parallelization; -1 turns on 
        self.model_configs = None
        self.best_models = None
        self.results_df = None
        self.roc_data = None
        self.misclassified_samples = None
        self.confusion_matrices = None
        self.shap_cache = {}

    def split_data(self, X, y):
        """
        Split the data with specified training, validation, and test splits.

        Args:
        - X: The feature dataset.
        - y: The target variable.
        - final_test_size (float): Proportion of the dataset for the final evaluation.
        - base_test_size (float): Proportion of the dataset for testing base models.
        - meta_validation_size (float): Proportion of the dataset for training the meta model.
        """
        # Initial split: Separate out the final evaluation set
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Separate out the validation and train sets
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(self.val_size / (1 - self.test_size)), random_state=self.random_state)
        
        # Scaling numerical features
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']  
        X_train_numerical = self.scaler.fit_transform(X_train[numerical_cols])
        X_val_numerical = self.scaler.transform(X_val[numerical_cols])
        X_test_numerical = self.scaler.transform(X_test[numerical_cols])
        
        # Combine scaled numerical features and one-hot encoded features
        X_train_scaled = pd.DataFrame(X_train_numerical, columns=numerical_cols, index=X_train.index).join(X_train.drop(columns=numerical_cols)).reset_index(drop=True)
        X_val_scaled = pd.DataFrame(X_val_numerical, columns=numerical_cols, index=X_val.index).join(X_val.drop(columns=numerical_cols)).reset_index(drop=True)
        X_test_scaled = pd.DataFrame(X_test_numerical, columns=numerical_cols, index=X_test.index).join(X_test.drop(columns=numerical_cols)).reset_index(drop=True)

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def generate_model_variants(self):
        """
        Prepares model configurations for use with GridSearchCV.
        """

        self.model_configs = {
            'Random Forest': {
                'model_class': RandomForestClassifier,
                'param_grid': {'n_estimators': [10, 50, 100, 150, 200]}
            },
            'XGBoost': {
                'model_class': XGBClassifier,
                'param_grid': {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'n_estimators': [100],
                    'use_label_encoder': [False],
                    'eval_metric': ['logloss']
                }
            },
            'MLP': {
                'model_class': MLPClassifier,
                'param_grid': {
                    'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
                    'learning_rate_init': [0.0005, 0.0001]
                }
            }
        }
    
    def select_best_variant(self, X_train, y_train, n_splits=5):
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        best_models = {}

        for model_name, config in self.model_configs.items():
            # print(f"\nEvaluating Baseline {model_name} variants on train set...")

            grid_search = GridSearchCV(
                estimator=config['model_class'](random_state=self.random_state),
                param_grid=config['param_grid'],
                scoring='f1',
                cv=kf,
                verbose=0, 
                n_jobs=self.n_jobs 
            )

            grid_search.fit(X_train, y_train)

            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            # print(f"Best variant for {model_name} found with Avg F1-Score {best_score:.4f} and params {best_params}")

            best_models[model_name] = {
                'model': best_model,
                'avg_f1_score': best_score,
                'params': best_params
            }

        self.best_models = best_models
    
    def _evaluate_model_performance(self, model, X_test, y_test):
        """
        Helper function to evaluate the performance of a model on the test set.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        roc_curve_data = {'fpr': None, 'tpr': None, 'roc_auc': None}
        pr_curve_data = {'precision': None, 'recall': None, 'pr_auc': None}

        if y_proba is not None:
            try:
                # Compute ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

                # Compute Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall, precision)  # AUC for PR is Recall -> Precision
                pr_curve_data = {'precision': precision, 'recall': recall, 'pr_auc': pr_auc}
            
            except Exception as e:
                print(f"Warning: PR AUC computation failed for model {model}. Error: {e}")
                pr_auc = None  # Ensure it's explicitly set

        else:
            print(f"Warning: {model} does not support predict_proba(), setting PR AUC to None.")
            pr_auc = None  # Set explicitly if model does not support probability outputs

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc if y_proba is not None else None,
            'PR AUC': pr_auc  # Ensure PR AUC is always defined
        }

        conf_matrix = confusion_matrix(y_test, y_pred)
        misclassified_indices = np.where(y_test != y_pred)[0]

        return {
            'metrics': metrics,
            'roc_curve_data': roc_curve_data,
            'pr_curve_data': pr_curve_data,  # Ensure PR curve data is passed along
            'conf_matrix': conf_matrix,
            'misclassified_indices': misclassified_indices
        }
  
    def evaluate_best_variant_on_validation(self, X_train, y_train, X_val, y_val):
        """
        Evaluate the best model variants on the validation set and store results.
        
        This method iterates over the best model variants, calls `evaluate_model_performance` for each,
        and then stores the results in the respective class attributes.
        
        Args:
        - X_train: Training set features.
        - y_train: Training set labels.
        - X_val: Validation set features.
        - y_val: Validation set labels.
        """
        performance_data = []

        for model_name, model_info in self.best_models.items():
            model = model_info['model']
            # model.fit(X_train, y_train)
            
            evaluation = self._evaluate_model_performance(model, X_val, y_val)
            evaluation['Model'] = model_name
            performance_data.append(evaluation)

        self.misclassified_samples = {
            data['Model']: {'indices': data['misclassified_indices']} for data in performance_data
        }
    
    def evaluate_base_models_on_test(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the best model variants on the test set and return results.
        
        This method iterates over the best model variants, calls `_evaluate_model_performance` for each,
        and then stores the results in the respective class attributes.
        
        Args:
        - X_train: Training set features.
        - y_train: Training set labels.
        - X_test: Test set features.
        - y_test: Test set labels.
        """

        performance_data = {}

        for model_name, model_info in self.best_models.items():
            model = model_info['model']
            # model.fit(X_train, y_train)
            performance_data[model_name] = self._evaluate_model_performance(model, X_test, y_test)

        return performance_data
    
    def _extract_best_models(self):
        """
        Extracts the 'model' attribute from each entry in the best_models dictionary. 
        """
        models = {}
        for model_name, model_info in self.best_models.items():
            models[model_name] = model_info['model']
        return models
    
    def _extract_feature_importance(self):
        """
        Extracts the XGBoost feature importance from the best variant baseline model.  
        """
        best_model_instances = self._extract_best_models()
        feature_importances = best_model_instances['XGBoost'].feature_importances_
        return feature_importances
 
    def _extract_unique_misclassified_samples(self, X_val):
        """
        Extracts the unique misclassified samples of each baseline model from X_val into a dictionary for easier comparison
        with samples from X_test. 
        """
        # Use set intersection to find common misclassified samples across all models
        common_misclassified_indices = set(self.misclassified_samples[next(iter(self.misclassified_samples))]['indices'])
        for model in self.misclassified_samples:
            common_misclassified_indices.intersection_update(self.misclassified_samples[model]['indices'])

        # Remove the common misclassified indices from each model's misclassified list
        unique_samples = {}
        for model in self.misclassified_samples:
            unique_indices = set(self.misclassified_samples[model]['indices']) - common_misclassified_indices
            unique_samples[model] = {'indices': np.array(list(unique_indices))}

        # Use unique lists of misclassified sample indices to retrieve the correct data from the first test set
        misclassified_dict = {}
        for model in unique_samples.keys():
            misclassified_dict[model] = X_val.iloc[unique_samples[model]['indices']]
        
        return misclassified_dict

    def _calculate_weighted_median_distances(self, X_val, X_test):
        """
        Calculate the Euclidean distances from each sample in the test set to the median of
        misclassified samples, weighted by feature importances.

        Returns:
            dict: A dictionary with model names as keys and a list of distances for each test sample as values.
        """

        feature_importances = self._extract_feature_importance()
        misclassified_dict = self._extract_unique_misclassified_samples(X_val)

        weighted_distance_dict = {model: [] for model in misclassified_dict}

        # Calculate the median for each model's misclassified samples
        misclassified_medians = {model: features.median(axis=0) for model, features in misclassified_dict.items()}

        # Iterate over each test sample
        for _, sample in X_test.iterrows():
            # Reshape the sample to be a 2D array for distance calculation
            sample_array = sample.values.reshape(1, -1)

            # Calculate weighted distances to medians
            for model, median in misclassified_medians.items():
                # Ensure median is a 2D array
                median_array = median.values.reshape(1, -1) * feature_importances
                sample_array_weighted = sample_array * feature_importances
                sample_array_weighted = np.array(sample_array_weighted, dtype=float)
                
                # Calculate the weighted distance
                weighted_distance = distance.cdist(sample_array_weighted, median_array, 'euclidean')[0]
                
                # Append the distance to the corresponding model list in weighted_distance_dict
                weighted_distance_dict[model].append(weighted_distance[0])

        return weighted_distance_dict
    
    def _calculate_weights_from_distances(self, distances_dict):
        """
        Calculate normalized weights from distances for each test sample.

        Args:
            distances_dict (dict): A dictionary where keys are model names and values are lists of distances for each test sample.

        Returns:
            dict: A dictionary with normalized weights for each model, corresponding to each test sample.
        """
        # Initialize a dictionary to store the normalized weights for each model
        normalized_weights_dict = {model: [] for model in distances_dict}
        
        # Get the total number of samples
        num_samples = len(next(iter(distances_dict.values())))
        
        # Normalize weights for each sample across all models
        for i in range(num_samples):
            # Gather distances for sample i from each model
            sample_distances = np.array([distances_dict[model][i] for model in distances_dict])
            # Normalize weights so they sum to 1
            normalized_weights = sample_distances / np.sum(sample_distances)
            # Assign normalized weights back to each model for sample i
            for j, model in enumerate(distances_dict):
                normalized_weights_dict[model].append(normalized_weights[j])
        
        return normalized_weights_dict

    def _weighted_pred_and_proba(self, X_test, weights):
        """
        Calculate the wighted ensemble predicted probabilities and predictions.

        Args:
        - X_test: Test set features.
        - weights: Dictionary of weights for each model.

        Returns:
        - Tuple of two elements: (ensemble_predictions, ensemble_probabilities)
        """
        best_model_instances = self._extract_best_models()

        # Extracts probabilities for each model that the new samples belong to class 1
        probas = {model: [] for model in best_model_instances}
        for model_name, model in best_model_instances.items():
            probas[model_name] = model.predict_proba(X_test)[:, 1]

        ensemble_predictions = []
        ensemble_probabilities = []

        for i in range(len(next(iter(probas.values())))):
            weighted_sum = 0

            # For each model, multiply its predicted probability by its weight for the current sample
            for model in probas:
                weighted_sum += probas[model][i] * weights[model][i]

            ensemble_probabilities.append(weighted_sum)

            # If the weighted sum is greater than 0.5, predict 1, otherwise predict 0
            prediction = 1 if weighted_sum > 0.5 else 0
            ensemble_predictions.append(prediction)

        return ensemble_predictions, ensemble_probabilities

    def _evaluate_weighted_ensemble_performance(self, y_pred, y_proba, y_test):
        """
        Evaluate the performance of the weighted ensemble model on the test set.
        """
        roc_curve_data = {'fpr': None, 'tpr': None, 'roc_auc': None}
        pr_curve_data = {'precision': None, 'recall': None, 'pr_auc': None}

        roc_auc, pr_auc = None, None

        if y_proba is not None:
            try:
                # Compute ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

                # Compute Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall, precision)  # PR AUC integrates Recall -> Precision
                pr_curve_data = {'precision': precision, 'recall': recall, 'pr_auc': pr_auc}
            
            except Exception as e:
                print(f"Warning: PR AUC computation failed for Weighted Ensemble. Error: {e}")

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc if y_proba is not None else None,
            'PR AUC': pr_auc if y_proba is not None else None
        }

        conf_matrix = confusion_matrix(y_test, y_pred)
        misclassified_indices = np.where(y_test != y_pred)[0]

        return {
            'metrics': metrics,
            'roc_curve_data': roc_curve_data,
            'pr_curve_data': pr_curve_data,  # Ensure PR data is included
            'conf_matrix': conf_matrix,
            'misclassified_indices': misclassified_indices
        }

    def _generate_meta_features(self, X):
        """
        Generate meta-features using the best variants of the base models.
        This function predicts with each base model and stacks the predictions.
        Args:
        - X: Data to generate meta-features for (either X_val_scaled or X_test_scaled).
        Returns:
        - A numpy array of meta-features.
        """
        meta_features = []
        for model_info in self.best_models.values():
            model = model_info['model']
            predictions = model.predict_proba(X)[:, 1] 
            meta_features.append(predictions)
        
        return np.column_stack(meta_features)

    def evaluate_meta_model_with_cv(self, X_val, y_val, X_test, y_test, n_splits=5):
        """
        Trains a meta-model with cross-validation and evaluates its performance.
        Args:
        - X_val, y_val: Validation features and labels for training the meta-model.
        - X_test, y_test: Test features and labels for evaluating the meta-model.
        - n_splits: Number of splits for cross-validation.
        Returns:
        - A dictionary with performance metrics and the trained meta-model.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs']
        }
        
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=10000),
            param_grid=param_grid,
            cv=kf,
            scoring='f1',
            verbose=0,
            n_jobs=self.n_jobs
        )
        
        # Generate meta-features for the validation set
        X_meta_val = self._generate_meta_features(X_val)
        
        # Perform the grid search
        grid_search.fit(X_meta_val, y_val)
        
        best_model = grid_search.best_estimator_
        
        # Generate meta-features for the test set
        X_meta_test = self._generate_meta_features(X_test)
        blended_ensemble_performance = self._evaluate_model_performance(best_model, X_meta_test, y_test)
        
        # Include the best model in the result dictionary for further analysis
        blended_ensemble_performance['model'] = best_model  # Add the best model to the performance dictionary
        
        return blended_ensemble_performance

    def run_weighted_ensemble(self, X_val, X_test, y_test):
        """
        Run the weighted ensemble on the test set and calculate performance.
        """
        # print(f"Starting weighted ensemble run on random state {self.random_state}")

        # Calculate distances and weights for weighted ensemble
        distances_dict = self._calculate_weighted_median_distances(X_val, X_test)
        weights_dict = self._calculate_weights_from_distances(distances_dict)

        # Get predictions and probabilities for weighted ensemble
        weighted_ensemble_pred, weighted_ensemble_proba = self._weighted_pred_and_proba(X_test, weights_dict)
        weighted_ensemble_performance = self._evaluate_weighted_ensemble_performance(weighted_ensemble_pred, weighted_ensemble_proba, y_test)

        # Initialize weighted SHAP values with the shape of the first base model's SHAP values
        first_model_name, first_model_info = list(self.best_models.items())[0]
        first_model = first_model_info['model']
        sample_shap_values = self.calculate_shap_values(first_model, X_val, first_model_name)
        
        # Ensure `sample_shap_values` is 2-dimensional by averaging over classes if needed
        if sample_shap_values.ndim == 3:
            sample_shap_values = sample_shap_values.mean(axis=2)

        # Initialize the weighted SHAP values array with the correct shape
        weighted_shap_values = np.zeros_like(sample_shap_values)

        # Print statement to confirm the start of SHAP calculation accumulation
        # print(f"Starting SHAP calculation for weighted ensemble on random state {self.random_state}")

        # Iterate through base models to accumulate weighted SHAP values
        for model_name, model_info in self.best_models.items():
            # print(f"Calculating and accumulating SHAP for model: {model_name} in weighted ensemble")
            model = model_info['model']
            
            # Retrieve SHAP values from cache (no redundant calculations)
            shap_values = self.calculate_shap_values(model, X_val, model_name)
            weight = np.mean(weights_dict[model_name])  # Average weight across samples
            
            # Handle potential 3D SHAP output (e.g., multi-class SHAP values) by averaging over the last dimension
            if shap_values.ndim == 3:
                shap_values = shap_values.mean(axis=2)  # Average across classes
            
            # Accumulate the weighted SHAP values
            weighted_shap_values += weight * shap_values

        # Print final weighted SHAP values to confirm they have been accumulated correctly
        # print(f"Final weighted SHAP values for random state {self.random_state}: {weighted_shap_values}")

        return weighted_ensemble_performance, weighted_shap_values

    def calculate_shap_values(self, model, X, model_name):
        """
        Calculate SHAP values with caching to avoid redundant computations.
        
        Args:
            model: Trained model for which to calculate SHAP values.
            X: Feature data to use for SHAP calculations.
            model_name: Name of the model to check in the cache.
            
        Returns:
            shap_values: SHAP values calculated for the given model and data.
        """
        if model_name in self.shap_cache:
            return self.shap_cache[model_name]
        
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X)

        # print(f"Calculating SHAP for model: {model} on random state {self.random_state}")
        shap_values = explainer.shap_values(X)
        
        self.shap_cache[model_name] = shap_values  # Cache the result
        return shap_values

    def run_models(self, X, y, n_splits=5):
        """
        Run the pipeline for training and evaluating the base models, weighted ensemble, and blended ensemble.
        
        Args:
        - X, y: Features and target variable.
        - n_splits: Number of splits for cross-validation in the meta model.
        
        Returns:
        - A dictionary containing the performance metrics and SHAP values for each ensemble model and base models.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Generate model variants and select best model variant
        self.generate_model_variants()
        self.select_best_variant(X_train, y_train, n_splits=n_splits)
        self.evaluate_best_variant_on_validation(X_train, y_train, X_val, y_val)
        base_performance = self.evaluate_base_models_on_test(X_train, y_train, X_test, y_test)

        # Initialize a dictionary to store SHAP values for all models
        shap_values_dict = {"base_models": {}}

        # Calculate and cache SHAP values for each base model only once per random state
        for model_name, model_info in self.best_models.items():
            model = model_info['model']
            shap_values = self.calculate_shap_values(model, X_val, model_name)  # Pass model_name for caching
            shap_values_dict["base_models"][model_name] = shap_values  # Store in dictionary

        # Run weighted ensemble and calculate SHAP values
        weighted_ensemble_performance, weighted_shap_values = self.run_weighted_ensemble(X_val, X_test, y_test)
        shap_values_dict["weighted_ensemble"] = weighted_shap_values

        # Run blended ensemble (meta model) and calculate SHAP values for the meta model
        blended_ensemble_performance = self.evaluate_meta_model_with_cv(X_val, y_val, X_test, y_test, n_splits=n_splits)
        blended_model = blended_ensemble_performance['model']  # Get the best meta-model

        # Generate meta-features for validation set
        X_meta_val = self._generate_meta_features(X_val)
        blended_shap_values = self.calculate_shap_values(blended_model, X_meta_val, "blended_ensemble")  # SHAP for blended ensemble
        shap_values_dict["blended_ensemble"] = blended_shap_values

        return {
            "weighted_ensemble_performance": weighted_ensemble_performance,
            "blended_ensemble_performance": blended_ensemble_performance,
            "base_performance": base_performance,
            "shap_values": shap_values_dict  # Include SHAP values in the output
        }

    def run_pipeline(self, random_states, df):
        """
        Run pipeline for each state in random_states and store results in dictionaries.
        """
        # Preprocess data by creating dummy variables
        dummy_df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'FastingBS', 
                                            'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
        X = dummy_df.drop('HeartDisease', axis=1)
        y = dummy_df['HeartDisease']

        # Initialize the SHAP values dictionary with the required structure
        shap_values_aggregated = {
            "weighted_ensemble": {},   # Initialize as empty dictionary for each state
            "blended_ensemble": {},
            "base_models": {}
        }

        # Dictionaries to store performance metrics
        weighted_ensemble_dict = {}
        blended_ensemble_dict = {}
        baseline_dict = {}

        # Run pipeline for each random state
        for state in random_states:
            # Reset the random state and clear SHAP cache for each iteration
            self.random_state = state
            self.shap_cache = {}  # Clear the SHAP cache for each random state

            # Run the models for this random state, capturing performance and SHAP values
            results = self.run_models(X, y)

            # Store performance metrics in dictionaries
            weighted_ensemble_dict[state] = results["weighted_ensemble_performance"]
            blended_ensemble_dict[state] = results["blended_ensemble_performance"]
            baseline_dict[state] = results["base_performance"]

            # Store SHAP values in aggregated dictionary
            shap_values = results["shap_values"]
            shap_values_aggregated["weighted_ensemble"][state] = shap_values.get("weighted_ensemble")
            shap_values_aggregated["blended_ensemble"][state] = shap_values.get("blended_ensemble")
            
            for model_name, shap_val in shap_values["base_models"].items():
                if model_name not in shap_values_aggregated["base_models"]:
                    shap_values_aggregated["base_models"][model_name] = {}
                shap_values_aggregated["base_models"][model_name][state] = shap_val

        return weighted_ensemble_dict, blended_ensemble_dict, baseline_dict, shap_values_aggregated

    def consolidate_performance_metrics(self, weighted_ensemble_dict, blended_ensemble_dict, baseline_dict):
        """
        Consolidates performance metrics across random states for each model into a DataFrame.
        """
        data_list = []

        # Assuming the same random states are available across all dictionaries
        random_states = weighted_ensemble_dict.keys()
        
        for random_state in random_states:
            # Add metrics for the weighted ensemble
            for model_name, model_dict in [
                ('Weighted Ensemble', weighted_ensemble_dict),
                ('Blended Ensemble', blended_ensemble_dict)
            ]:
                metrics = model_dict[random_state]['metrics']
                data_list.append({
                    'Random State': random_state,
                    'Model': model_name,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1-Score': metrics['F1-Score'],
                    'ROC AUC': metrics['ROC AUC'],
                    'PR AUC': metrics.get('PR AUC', None)  # Fix: Ensure no KeyError
                })

            # Add metrics for each baseline model
            for model_name in baseline_dict[random_state]:
                metrics = baseline_dict[random_state][model_name]['metrics']
                data_list.append({
                    'Random State': random_state,
                    'Model': model_name.replace('_', ' '),
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1-Score': metrics['F1-Score'],
                    'ROC AUC': metrics['ROC AUC'],
                    'PR AUC': metrics.get('PR AUC', None)  # Fix: Ensure no KeyError
                })

        # Convert the list of dictionaries to a pandas DataFrame
        return pd.DataFrame(data_list)

    def aggregate_confusion_and_roc(self, weighted_ensemble_dict, blended_ensemble_dict, baseline_dict):
        """
        Aggregates confusion matrices, ROC curves, and PR curves across random states for each model.
        """
        aggregation = {
            'Model': [],
            'Random State': [],
            'Confusion Matrix': [],
            'FPR': [],
            'TPR': [],
            'ROC AUC': [],
            'Precision': [],
            'Recall': [],
            'PR AUC': []
        }

        for random_state in weighted_ensemble_dict.keys():
            for model, data_dict in [('Weighted Ensemble', weighted_ensemble_dict),
                                    ('Blended Ensemble', blended_ensemble_dict)]:
                aggregation['Model'].append(model)
                aggregation['Random State'].append(random_state)
                aggregation['Confusion Matrix'].append(data_dict[random_state]['conf_matrix'])
                aggregation['FPR'].append(data_dict[random_state]['roc_curve_data']['fpr'])
                aggregation['TPR'].append(data_dict[random_state]['roc_curve_data']['tpr'])
                aggregation['ROC AUC'].append(data_dict[random_state]['roc_curve_data']['roc_auc'])
                aggregation['Precision'].append(data_dict[random_state]['pr_curve_data']['precision'])
                aggregation['Recall'].append(data_dict[random_state]['pr_curve_data']['recall'])
                aggregation['PR AUC'].append(data_dict[random_state]['pr_curve_data']['pr_auc'])

            for model_name in baseline_dict[random_state]:
                aggregation['Model'].append(model_name)
                aggregation['Random State'].append(random_state)
                aggregation['Confusion Matrix'].append(baseline_dict[random_state][model_name]['conf_matrix'])
                aggregation['FPR'].append(baseline_dict[random_state][model_name]['roc_curve_data']['fpr'])
                aggregation['TPR'].append(baseline_dict[random_state][model_name]['roc_curve_data']['tpr'])
                aggregation['ROC AUC'].append(baseline_dict[random_state][model_name]['roc_curve_data']['roc_auc'])
                aggregation['Precision'].append(baseline_dict[random_state][model_name]['pr_curve_data']['precision'])
                aggregation['Recall'].append(baseline_dict[random_state][model_name]['pr_curve_data']['recall'])
                aggregation['PR AUC'].append(baseline_dict[random_state][model_name]['pr_curve_data']['pr_auc'])

        return pd.DataFrame(aggregation)

    def plot_median_confusion_matrices_grid(self, df):
        """
        Plot median confusion matrices for each model in a grid with integer-adjusted values.

        Parameters:
        - df: Pandas DataFrame containing columns ['Model', 'Confusion Matrix']
        """
        models = df['Model'].unique()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
        axes = axes.flatten()
        class_names = ['No Heart Disease', 'Heart Disease']
        
        total_samples = int(np.sum(df.iloc[0]['Confusion Matrix']))  # Ensure sum matches test set size

        for i, model in enumerate(models[:5]):
            confusion_matrices = df[df['Model'] == model]['Confusion Matrix']
            matrices_array = np.array([np.array(cm) for cm in confusion_matrices])

            # Compute the median confusion matrix (floating point)
            median_conf_matrix = np.median(matrices_array, axis=0)

            # Scale matrix to ensure it sums to total_samples
            scaled_conf_matrix = (median_conf_matrix / np.sum(median_conf_matrix)) * total_samples

            # Convert to integers while preserving sum
            final_conf_matrix = np.round(scaled_conf_matrix).astype(int)

            # Ensure sum remains exactly total_samples by adjusting the last value
            discrepancy = total_samples - np.sum(final_conf_matrix)
            final_conf_matrix[np.unravel_index(np.argmax(final_conf_matrix), final_conf_matrix.shape)] += discrepancy

            sns.heatmap(final_conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False, ax=axes[i],
                        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
            axes[i].set_title(f'Median Confusion Matrix for {model}')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')

        if len(models) < 6:
            axes[5].axis('off')  # Hide any unused subplot
        plt.tight_layout()
        plt.show()

    def plot_median_roc_curve(self, df, num_points=100):
        """
        Plot a median ROC curve across random states for each model.
        """
        plt.figure(figsize=(8, 6), dpi=300)
        mean_fpr = np.linspace(0, 1, num_points)
        
        for model in df['Model'].unique():
            all_tpr = []
            
            for _, row in df[df['Model'] == model].iterrows():
                fpr = np.array(row['FPR'])
                tpr = np.array(row['TPR'])
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                all_tpr.append(interp_tpr)
            
            median_tpr = np.median(all_tpr, axis=0)
            plt.plot(mean_fpr, median_tpr, label=f'{model} (Median ROC AUC: {df[df["Model"] == model]["ROC AUC"].median():.4f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Median ROC Curve Across Random States')
        plt.legend(loc='lower right')
        plt.show()

    def plot_median_pr_curve(self, df, num_points=100):
        """
        Plot a median Precision-Recall (PR) curve across random states for each model.
        """
        plt.figure(figsize=(8, 6), dpi=300)
        mean_recall = np.linspace(0, 1, num_points)
        
        for model in df['Model'].unique():
            all_precision = []
            
            for _, row in df[df['Model'] == model].iterrows():
                recall = np.array(row['Recall'])
                precision = np.array(row['Precision'])
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])  # Interpolate for consistent recall points
                all_precision.append(interp_precision)
            
            median_precision = np.median(all_precision, axis=0)
            plt.plot(mean_recall, median_precision, label=f'{model} (Median PR AUC: {df[df["Model"] == model]["PR AUC"].median():.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Median Precision-Recall Curve Across Random States')
        plt.legend(loc='lower left')
        plt.show()

    def aggregate_shap_values(self, shap_values_aggregated):
        """
        Aggregates SHAP values across random states for each model type using the median.
        
        Args:
        - shap_values_aggregated: Dictionary containing SHAP values for each random state and model type.

        Returns:
        - Dictionary of median SHAP values aggregated across random states for each model.
        """
        aggregated_shap = {
            "weighted_ensemble": None,
            "blended_ensemble": None,
            "base_models": {}
        }
        
        # Aggregate SHAP values for weighted and blended ensembles if they exist
        for model_type in ["weighted_ensemble", "blended_ensemble"]:
            shap_values_list = [shap_values_aggregated[model_type][state] 
                                for state in shap_values_aggregated[model_type] 
                                if shap_values_aggregated[model_type][state] is not None]
            if shap_values_list:  # Only calculate median if there’s at least one non-None entry
                aggregated_shap[model_type] = np.median(shap_values_list, axis=0)

        # Aggregate SHAP values for each base model
        for model_name in shap_values_aggregated["base_models"]:
            shap_values_list = [shap_values_aggregated["base_models"][model_name][state] 
                                for state in shap_values_aggregated["base_models"][model_name] 
                                if shap_values_aggregated["base_models"][model_name][state] is not None]
            if shap_values_list:
                aggregated_shap["base_models"][model_name] = np.median(shap_values_list, axis=0)
        
        return aggregated_shap

    def plot_shap_summary(self, aggregated_shap, feature_names, model_type, max_display=10):
        """
        Plots a SHAP summary plot for the most important features of a specified model type.
        
        Args:
        - aggregated_shap: Dictionary of aggregated SHAP values across random states.
        - feature_names: List of feature names.
        - model_type: The model type for which to plot the SHAP values. Options are "weighted_ensemble", "blended_ensemble", or a base model name.
        - max_display: Maximum number of features to display in the plot.
        """
        # Retrieve SHAP values for the specified model type
        if model_type in aggregated_shap:
            shap_values = aggregated_shap[model_type]
        elif model_type in aggregated_shap["base_models"]:
            shap_values = aggregated_shap["base_models"][model_type]
        else:
            raise ValueError(f"Invalid model type specified: {model_type}")
        
        # Set meta feature names for the blended ensemble
        if model_type == "blended_ensemble":
            feature_names = [f"{name} Prediction" for name in self.best_models.keys()]
        
        # Ensure `shap_values` is an array of feature contributions and handle 3D shape
        if shap_values.ndim == 3:
            # Select the positive class (class 1) SHAP values
            shap_values = shap_values[:, :, 1]
        
        # Compute median absolute SHAP values across samples and features
        median_abs_shap = np.median(np.abs(shap_values), axis=0)  # median across samples
        
        # Adjust `feature_names` and `median_abs_shap` to match each other’s length if there’s a mismatch
        if len(median_abs_shap) != len(feature_names):
            min_len = min(len(median_abs_shap), len(feature_names))
            median_abs_shap = median_abs_shap[:min_len]
            feature_names = feature_names[:min_len]

        # Adjust max_display to avoid index out-of-bounds errors
        max_display = min(max_display, len(median_abs_shap))

        # Get indices for the top features by median absolute SHAP value
        top_indices = np.argsort(median_abs_shap)[-max_display:]
        top_features = [feature_names[i] for i in top_indices]
        
        # Plot the SHAP summary
        plt.figure(figsize=(10, 6), dpi=300)
        plt.barh(range(len(top_indices)), median_abs_shap[top_indices], align="center")
        plt.yticks(range(len(top_indices)), top_features)
        plt.xlabel("Median |SHAP Value|")
        plt.title(f"Top {len(top_indices)} Features by SHAP Value for {model_type}")
        plt.gca().invert_yaxis()
        plt.show()
