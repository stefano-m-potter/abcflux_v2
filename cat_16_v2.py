#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

def run_loso_analysis(target_col):
    """
    Performs a full Leave-One-Site-Out (LOSO) cross-validation for a given target variable.
    This function is self-contained, handling its own data loading and using a fixed
    set of hyperparameters for the model.

    Args:
        target_col (str): The name of the target variable column (e.g., 'gpp', 'nee').
    """
    # 1. Load and prepare dataset
    print(f"--- Processing Target: {target_col.upper()} ---")
    file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final_mod16.csv"
    df = pd.read_csv(file_path)
    df['land_cover'] = df['land_cover'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[df['flux_method'] == 'EC']

    # 2. Create derived features
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # 3. Define predictors and categorical features
    feature_cols = [
        'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03',
        'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
        'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
        'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
        'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
        'land_cover', 'month',
        'lai', 'fpar', 'Percent_NonTree_Vegetation',
        'Percent_NonVegetated', 'Percent_Tree_Cover'
    ]
    categorical_features = ['land_cover', 'month']

    # Drop rows where the current target or site_reference is missing
    df = df.dropna(subset=['site_reference', target_col])

    # 4. Define output paths
    loocv_out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv_16", target_col)
    figures_path = os.path.join(loocv_out_path, "figures")
    models_out_path = '/explore/nobackup/people/spotter5/anna_v/v2/models_16'
    os.makedirs(loocv_out_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_out_path, exist_ok=True)

    # 5. Prepare features (X) and target (y)
    X = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    for col in categorical_features:
        X[col] = X[col].astype('category')

    results = []
    all_preds_df_list = []

    # 6. Leave-One-Site-Out CV Loop
    for test_site in sites:
        print(f"  Processing site: {test_site}...")
        train_idx = df["site_reference"] != test_site
        test_idx = df["site_reference"] == test_site

        if test_idx.sum() < 1:
            continue

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        dates_test = df.loc[test_idx, "date"]

        # Initialize and train the CatBoost model with a fixed set of parameters
        model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.01,
            depth=8,
            subsample=0.7,
            random_state=42,
            l2_leaf_reg=0.1,
            rsm=0.8,
            cat_features=categorical_features,
            verbose=0,
            allow_writing_files=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect predictions and metrics
        site_df = pd.DataFrame({"Site": test_site, "Date": dates_test.values, "Observed": y_test.values, "Predicted": y_pred})
        all_preds_df_list.append(site_df)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Site": test_site, "RMSE": rmse, "MAE": mae, "R2": r2})

    if not results:
        print(f"No data processed for target '{target_col}'. Skipping.")
        return

    # 7. Combine, Save, and Report Results
    results_df = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

    results_csv_path = os.path.join(loocv_out_path, f'catboost_results_{target_col}_cat.csv')
    predictions_csv_path = os.path.join(loocv_out_path, f'catboost_predictions_{target_col}_cat.csv')
    results_df.to_csv(results_csv_path, index=False)
    all_preds_df.to_csv(predictions_csv_path, index=False)
    print(f"\n  Results saved to: {results_csv_path}")

    # Report pooled metrics
    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    print(f"\n  --- Pooled Metrics for {target_col.upper()} ---")
    print(f"  Pooled R²: {r2_all:.4f}, Pooled RMSE: {rmse_all:.4f}, Pooled MAE: {mae_all:.4f}")

    # --- START: EDITED SECTION ---
    # Calculate and report mean/median metrics across all sites
    mean_r2 = results_df['R2'].mean()
    median_r2 = results_df['R2'].median()
    mean_rmse = results_df['RMSE'].mean()
    median_rmse = results_df['RMSE'].median()
    mean_mae = results_df['MAE'].mean()
    median_mae = results_df['MAE'].median()
    
    print(f"\n  --- Summary Metrics Across Sites for {target_col.upper()} ---")
    print(f"  Mean R²:   {mean_r2:.4f}, Median R²:   {median_r2:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f}, Median RMSE: {median_rmse:.4f}")
    print(f"  Mean MAE:  {mean_mae:.4f}, Median MAE:  {median_mae:.4f}")
    # --- END: EDITED SECTION ---

    # 8. Plotting
    print("\n  Generating and saving individual site plots...")
    for site in all_preds_df["Site"].unique():
        fig, ax = plt.subplots(figsize=(12, 7))
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.legend(), ax.grid(True), fig.autofmt_xdate()

        textstr = f"RMSE: {site_metrics['RMSE']:.2f}\nMAE: {site_metrics['MAE']:.2f}\nR²: {site_metrics['R2']:.2f}"
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        plot_filename = f'catboost_{target_col}_{site}_timeseries_cat.png'
        plot_path = os.path.join(figures_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  All site plots saved to: {figures_path}")

    # 9. Train and Save Final Model on all data for the target
    print("\n  Training and saving final model on all data...")
    final_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.01,
        depth=8,
        subsample=0.7,
        random_state=42,
        l2_leaf_reg=0.1,
        rsm=0.8,
        cat_features=categorical_features,
        verbose=0,
        allow_writing_files=False
    )
    final_model.fit(X, y)
    model_filename = os.path.join(models_out_path, f'{target_col}.json')
    final_model.save_model(model_filename)
    print(f"  Final model saved to: {model_filename}")


if __name__ == '__main__':
    # List of target variables to run the analysis for
    targets_to_run = ['gpp', 'nee', 'reco', 'ch4_flux_total']

    # Loop through each target and run the full analysis
    for target in targets_to_run:
        print(f"\n{'='*50}\nRUNNING ANALYSIS FOR: {target.upper()}\n{'='*50}")
        run_loso_analysis(target_col=target)
        print(f"\n{'='*50}\nCOMPLETED ANALYSIS FOR: {target.upper()}\n{'='*50}")