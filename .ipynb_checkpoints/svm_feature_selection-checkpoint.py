#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

def run_loso_with_top_features(target_col):
    """
    Performs Leave-One-Site-Out (LOSO) modeling using a Radial Support Vector Machine (SVM).
    Features are one-hot encoded and standardized within a pipeline.
    """
    # --- 1. Load the file with the top pre-selected features ---
    base_path = "/explore/nobackup/people/spotter5/anna_v/v2/loocv"
    top_features_path = os.path.join(base_path, target_col, f'training_data_{target_col}_top_preds.csv')

    try:
        # Load the CSV with top predictors to get the feature list
        top_features_df = pd.read_csv(top_features_path)
    except FileNotFoundError:
        print(f"SKIPPING: Top features file not found for target '{target_col}'.")
        print(f"Expected at: {top_features_path}")
        print("Please run the feature selection script for this target first.\n")
        return

    # Automatically get the feature columns from the loaded file
    feature_cols = [col for col in top_features_df.columns if col != target_col]
    print(f"Using top selected features for '{target_col}': {feature_cols}")

    # --- 2. Load the main dataset and prepare it ---
    full_dataset_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final.csv"
    df = pd.read_csv(full_dataset_path)

    # Basic data cleaning and preparation
    df['land_cover'] = df['land_cover'].astype('category')
    df['month'] = df['month'].astype('category')
    df = df[df['flux_method'] == 'EC']
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # Drop rows with missing data for essential columns
    required_cols = feature_cols + [target_col, 'site_reference']
    df = df.dropna(subset=required_cols)

    # --- 3. Prepare data for modeling (including one-hot encoding) ---
    out_path = os.path.join(base_path, target_col)
    figures_path = os.path.join(out_path, "figures_svm_top_features") # New folder for SVM plots
    os.makedirs(figures_path, exist_ok=True)

    X_initial = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    # One-hot encode categorical features. This must be done before the CV split.
    categorical_features = [f for f in X_initial.select_dtypes(include=['category', 'object']).columns if f in feature_cols]
    if categorical_features:
        print(f"One-hot encoding categorical features: {categorical_features}")
        X = pd.get_dummies(X_initial, columns=categorical_features, drop_first=True)
    else:
        X = X_initial
    
    # Update feature_cols to reflect the new dummy columns
    final_feature_cols = X.columns.tolist()

    results = []
    all_preds_df_list = []

    # --- 4. Run Leave-One-Site-Out CV with SVM ---
    for test_site in sites:
        print(f"  Processing site: {test_site}...")
        train_idx = df["site_reference"] != test_site
        test_idx = df["site_reference"] == test_site

        if test_idx.sum() < 1:
            continue

        # Split data based on site
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        dates_test = df.loc[test_idx, "date"]
        
        # Create a pipeline to standardize data and then apply SVM
        # This prevents data leakage from the test set into the training scaler
        model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        site_df = pd.DataFrame({
            "Site": test_site, "Date": dates_test.values,
            "Observed": y_test.values, "Predicted": y_pred
        })
        all_preds_df_list.append(site_df)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Site": test_site, "RMSE": rmse, "MAE": mae, "R2": r2})

    # --- 5. Aggregate, Save, and Report Results ---
    if not results:
        print(f"No data processed for target '{target_col}'.")
        return

    results_df = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

    # Update filenames to reflect the use of SVM
    results_csv_path = os.path.join(out_path, f'svm_results_{target_col}_top_features.csv')
    predictions_csv_path = os.path.join(out_path, f'svm_predictions_{target_col}_top_features.csv')
    results_df.to_csv(results_csv_path, index=False)
    all_preds_df.to_csv(predictions_csv_path, index=False)
    print(f"  Results saved to: {results_csv_path}")

    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    print(f"\n--- SVM Pooled Metrics for {target_col.upper()} (Top Features) ---")
    print(f"Pooled R²: {r2_all:.4f}, Pooled RMSE: {rmse_all:.4f}, Pooled MAE: {mae_all:.4f}")

    # --- 6. Plotting ---
    print("  Generating and saving individual site plots...")
    for site in all_preds_df["Site"].unique():
        fig, ax = plt.subplots(figsize=(12, 7))
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} (SVM, Top Features) for Site: {site}")
        ax.legend(), ax.grid(True), fig.autofmt_xdate()

        textstr = f"RMSE: {site_metrics['RMSE']:.2f}\nMAE: {site_metrics['MAE']:.2f}\nR²: {site_metrics['R2']:.2f}"
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Update plot filenames for SVM
        plot_filename = f'svm_{target_col}_{site}_timeseries_top_features.png'
        plot_path = os.path.join(figures_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  All site plots saved to: {figures_path}")

if __name__ == '__main__':
    # List of target variables to run the analysis for
    targets_to_run = ['gpp', 'nee', 'reco', 'ch4_flux_total']
    
    for target in targets_to_run:
        print(f"\n{'='*20} RUNNING SVM ANALYSIS FOR: {target.upper()} {'='*20}")
        run_loso_with_top_features(target_col=target)
        print(f"{'='*20} COMPLETED SVM ANALYSIS FOR: {target.upper()} {'='*20}")