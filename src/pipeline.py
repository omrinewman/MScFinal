from ensemble import Ensembling 
import pandas as pd
import warnings
import random


def run_ensemble_pipeline(
    ensemble_class,
    data_path="../data/heart.csv",
    random_seed=500,
    num_random_states=5,
    state=0):
    """
    Runs the full ensemble pipeline with specified parameters.

    Parameters:
        ensemble_class (class): The Ensembling class to instantiate.
        data_path (str): Path to the CSV dataset.
        random_seed (int): Seed for reproducibility.
        num_random_states (int): Number of random states to generate.
        state (int): Initial state for the ensemble.
    """
    if num_random_states < 2:
        raise ValueError("num_random_states must be at least 2.")

    warnings.filterwarnings("ignore")
    random.seed(random_seed)
    
    # Initialize ensemble
    ensemble = ensemble_class(state=state)
    
    # Generate random states
    random_states = [random.randint(1, 200) for _ in range(num_random_states)]
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Run pipeline
    weighted_ensemble_dict, blended_ensemble_dict, baseline_dict, shap_values_aggregated = (
        ensemble.run_pipeline(random_states, df)
    )
    
    # Consolidate performance metrics
    performance_metrics_df = ensemble.consolidate_performance_metrics(
        weighted_ensemble_dict=weighted_ensemble_dict,
        blended_ensemble_dict=blended_ensemble_dict,
        baseline_dict=baseline_dict
    )

    # print(performance_metrics_df)
    
    # Aggregate confusion matrices and ROC data
    confusion_roc_df = ensemble.aggregate_confusion_and_roc(
        weighted_ensemble_dict=weighted_ensemble_dict,
        blended_ensemble_dict=blended_ensemble_dict,
        baseline_dict=baseline_dict
    )
    
    # Plot performance visualizations
    ensemble.plot_median_confusion_matrices_grid(confusion_roc_df)
    ensemble.plot_median_roc_curve(confusion_roc_df)
    ensemble.plot_median_pr_curve(confusion_roc_df)
    
    # Aggregate SHAP values
    aggregated_shap_values = ensemble.aggregate_shap_values(shap_values_aggregated)
    feature_names = df.drop(columns=['HeartDisease']).columns
    
    # Plot SHAP summaries
    ensemble.plot_shap_summary(aggregated_shap_values, feature_names, model_type="weighted_ensemble", max_display=10)
    ensemble.plot_shap_summary(aggregated_shap_values, feature_names, model_type="blended_ensemble", max_display=10)
    
    for model_name in aggregated_shap_values["base_models"]:
        print(f"\nPlotting SHAP summary for base model: {model_name}")
        ensemble.plot_shap_summary(aggregated_shap_values, feature_names, model_type=model_name, max_display=10)
    
    # Compute median performance metrics
    median_performance_df = performance_metrics_df.groupby('Model').median(numeric_only=True).drop('Random State', axis=1)
    median_performance_df.columns = [f"median_{col}" for col in median_performance_df.columns]
    median_performance_df = median_performance_df.reindex(
        ['MLP', 'XGBoost', 'Random Forest', 'Blended Ensemble', 'Weighted Ensemble']
    )
    
    return performance_metrics_df, median_performance_df
