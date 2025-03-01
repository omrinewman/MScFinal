# Ensembling Class Overview
The `Ensembling` class is designed for training, evaluating, and interpreting ensemble models for heart failure prediction. It supports multiple base models, builds weighted and blended ensemble models, and incorporates SHAP interpretability for feature importance analysis. The class automates data preprocessing, model training, performance evaluation, and SHAP analysis across multiple random states.

## Installation
To use this repository, ensure you have Python and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset
The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data). It includes 918 patient data observations from five different countries, original sources can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Usage
### Running the Pipeline
To execute the full pipeline across multiple random states:
```python
from pipeline import run_ensemble_pipeline
from ensemble import Ensembling

data_path = "../data/heart.csv"
random_seed = 500
num_random_states = 5

performance_metrics_df, median_performance_df = run_ensemble_pipeline(Ensembling, 
                                              data_path = data_path, 
                                              random_seed = random_seed, 
                                              num_random_states = num_random_states)

display(performance_metrics_df)
display(median_performance_df)

```

## Summary
The `Ensembling` class automates data preparation, model selection, evaluation, ensemble building, and interpretability. It provides robust model comparison and visualization tools, making it suitable for heart failure prediction analysis with multiple random states.

The `run_ensemble_pipeline` function initiates the Ensembling class and handles the performance consolidation and visualization of the ensemble learning method across random states. 

## Class Initialization (`__init__`)
- Initializes the class with parameters such as random state, validation/test split proportions, and model-related attributes.
- Sets up an empty SHAP cache for efficient interpretability calculations.

## Data Preprocessing
### `split_data(X, y)`
- Splits dataset into training, validation, and test sets.
- Scales numerical features using `StandardScaler` and applies one-hot encoding to categorical features.

## Model Preparation and Selection
### `generate_model_variants()`
- Defines hyperparameter grids for three base models: Random Forest, XGBoost, and MLP.

### `select_best_variant(X_train, y_train, n_splits=5)`
- Uses cross-validation and `GridSearchCV` to select the best hyperparameter configuration for each base model.

## Model Evaluation
### `_evaluate_model_performance(model, X_test, y_test)`
- Computes classification metrics, ROC, and Precision-Recall curves.
- Identifies misclassified samples.

### `evaluate_best_variant_on_validation(X_train, y_train, X_val, y_val)`
- Trains best models and evaluates them on the validation set.

### `evaluate_base_models_on_test(X_train, y_train, X_test, y_test)`
- Evaluates the best variants of base models on the test set and returns results.

## Weighted Ensemble Model
### `_extract_best_models()`
- Extracts the trained best model instances.

### `_extract_feature_importance()`
- Retrieves feature importances from the best XGBoost model.

### `_extract_unique_misclassified_samples(X_val)`
- Identifies misclassified samples unique to each model.

### `_calculate_weighted_median_distances(X_val, X_test)`
- Computes Euclidean distances from test samples to misclassified samples, weighted by feature importance.

### `_calculate_weights_from_distances(distances_dict)`
- Normalizes distance-based weights for the ensemble models.

### `_weighted_pred_and_proba(X_test, weights)`
- Generates weighted ensemble predictions using distance-based weights.

### `_evaluate_weighted_ensemble_performance(y_pred, y_proba, y_test)`
- Evaluates the weighted ensemble model and computes classification metrics.

### `run_weighted_ensemble(X_val, X_test, y_test)`
- Runs the weighted ensemble model, evaluates it, and computes SHAP values.

## Blended Ensemble (Meta-Model)
### `_generate_meta_features(X)`
- Stacks predictions from base models to create meta-features.

### `evaluate_meta_model_with_cv(X_val, y_val, X_test, y_test, n_splits=5)`
- Trains a logistic regression meta-model using cross-validation.
- Evaluates its performance on the test set.

## SHAP Analysis
### `calculate_shap_values(model, X, model_name)`
- Computes and caches SHAP values for a given model.

### `aggregate_shap_values(shap_values_aggregated)`
- Aggregates SHAP values across random states using the median.

### `plot_shap_summary(aggregated_shap, feature_names, model_type, max_display=10)`
- Visualizes the top ten most important features using SHAP values.

## Full Pipeline Execution
### `run_models(X, y, n_splits=5)`
- Runs base models, weighted and blended ensembles, and evaluates performance.

### `run_pipeline(random_states, df)`
- Runs the full pipeline across multiple random states and aggregates results.

## Performance Consolidation & Visualization
### `consolidate_performance_metrics(weighted_ensemble_dict, blended_ensemble_dict, baseline_dict)`
- Combines performance metrics from multiple random states into a DataFrame.

### `aggregate_confusion_and_roc(weighted_ensemble_dict, blended_ensemble_dict, baseline_dict)`
- Aggregates confusion matrices, ROC curves, and PR curves across random states.

### `plot_median_confusion_matrices_grid(df)`
- Plots median confusion matrices for each model in a grid format.

### `plot_median_roc_curve(df, num_points=100)`
- Plots the median ROC curve across random states.

### `plot_median_pr_curve(df, num_points=100)`
- Plots the median Precision-Recall curve across random states.


