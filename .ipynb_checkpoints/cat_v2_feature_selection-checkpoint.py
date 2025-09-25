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

def run_loso_with_top_features(target_col):
    """
    Performs Leave-One-Site-Out modeling using a pre-selected list of top features.
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
    df['land_cover'] = df['land_cover'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[df['flux_method'] == 'EC']
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # Drop rows with missing data for essential columns
    required_cols = feature_cols + [target_col, 'site_reference']
    df = df.dropna(subset=required_cols)

    # Determine which categorical features are actually in our top feature list
    potential_cat_features = ['land_cover', 'month']
    categorical_features = [f for f in potential_cat_features if f in feature_cols]

    # --- 3. Prepare for LOSO Cross-Validation ---
    out_path = os.path.join(base_path, target_col)
    figures_path = os.path.join(out_path, "figures_top_features")
    os.makedirs(figures_path, exist_ok=True)

    X = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    results = []
    all_preds_df_list = []

    # --- 4. Run Leave-One-Site-Out CV ---
    for test_site in sites:
        print(f"  Processing site: {test_site}...")
        train_idx = df["site_reference"] != test_site
        test_idx = df["site_reference"] == test_site

        if test_idx.sum() < 1:
            continue

        X_train, y_train = X.loc[train_idx, feature_cols], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx, feature_cols], y.loc[test_idx]
        dates_test = df.loc[test_idx, "date"]

        model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.01,
            depth=8,
            subsample=0.7,
            random_state=42,
            l2_leaf_reg=0.1,
            cat_features=categorical_features,
            verbose=0,
            allow_writing_files=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        site_df = pd.DataFrame({
            "Site": test_site,
            "Date": dates_test.values,
            "Observed": y_test.values,
            "Predicted": y_pred
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

    results_csv_path = os.path.join(out_path, f'catboost_results_{target_col}_top_features.csv')
    predictions_csv_path = os.path.join(out_path, f'catboost_predictions_{target_col}_top_features.csv')
    results_df.to_csv(results_csv_path, index=False)
    all_preds_df.to_csv(predictions_csv_path, index=False)
    print(f"  Results saved to: {results_csv_path}")

    # --- pooled metrics ---
    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    print(f"\n--- Pooled Metrics for {target_col.upper()} (Top Features) ---")
    print(f"Pooled R²: {r2_all:.4f}, Pooled RMSE: {rmse_all:.4f}, Pooled MAE: {mae_all:.4f}")

    # --- NEW: Summary metrics across sites (mean/median) ---
    mean_r2 = results_df['R2'].mean()
    median_r2 = results_df['R2'].median()
    mean_rmse = results_df['RMSE'].mean()
    median_rmse = results_df['RMSE'].median()
    mean_mae = results_df['MAE'].mean()
    median_mae = results_df['MAE'].median()
    print(f"\n--- Summary Metrics Across Sites for {target_col.upper()} (Top Features) ---")
    print(f"  Mean R²:   {mean_r2:.4f}, Median R²:   {median_r2:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f}, Median RMSE: {median_rmse:.4f}")
    print(f"  Mean MAE:  {mean_mae:.4f}, Median MAE:  {median_mae:.4f}")

    # --- 6. Plotting ---
    print("  Generating and saving individual site plots...")
    for site in all_preds_df["Site"].unique():
        fig, ax = plt.subplots(figsize=(12, 7))
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} (Top Features) for Site: {site}")
        ax.legend(), ax.grid(True), fig.autofmt_xdate()

        textstr = f"RMSE: {site_metrics['RMSE']:.2f}\nMAE: {site_metrics['MAE']:.2f}\nR²: {site_metrics['R2']:.2f}"
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        plot_filename = f'catboost_{target_col}_{site}_timeseries_top_features.png'
        plot_path = os.path.join(figures_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  All site plots saved to: {figures_path}")

    # --- 7. Pooled Observed vs Predicted density plot (Top Features) ---
    pooled = all_preds_df[['Observed', 'Predicted']].copy()
    pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna()

    if not pooled.empty:
        fig, ax = plt.subplots(figsize=(7, 7))

        # Axis limits with padding
        lo = np.nanmin([pooled['Observed'].min(), pooled['Predicted'].min()])
        hi = np.nanmax([pooled['Observed'].max(), pooled['Predicted'].max()])
        pad = 0.05 * (hi - lo if np.isfinite(hi - lo) and (hi - lo) > 0 else 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        # Density plot (hexbin) — darker red = higher density
        hb = ax.hexbin(
            pooled['Observed'],
            pooled['Predicted'],
            gridsize=80,
            cmap='Reds',
            bins='log',
            mincnt=1
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('log10(N points)')

        # 1:1 line in solid black
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='black', linewidth=1.5)

        ax.set_title(f'Observed vs Predicted (LOSO, Top Features) — {target_col}')
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.grid(True)

        # Round pooled metrics to 2 decimals in the lower-right corner
        annot = f"R² = {r2_all:.2f}\nRMSE = {rmse_all:.2f}\nMAE = {mae_all:.2f}"
        ax.text(
            0.97, 0.03, annot, transform=ax.transAxes,
            fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

        # Save density plot
        pooled_plot_path = os.path.join(
            figures_path, f'catboost_{target_col}_obs_vs_pred_all_sites_top_features.png'
        )
        plt.savefig(pooled_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Pooled Observed vs Predicted density plot saved to: {pooled_plot_path}")
    else:
        print("  Skipped pooled density plot: no valid Observed/Predicted pairs after cleaning.")


if __name__ == '__main__':
    # List of target variables to run the analysis for
    targets_to_run = ['gpp', 'nee', 'reco', 'ch4_flux_total']
    
    for target in targets_to_run:
        print(f"\n{'='*20} RUNNING ANALYSIS FOR: {target.upper()} {'='*20}")
        run_loso_with_top_features(target_col=target)
        print(f"{'='*20} COMPLETED ANALYSIS FOR: {target.upper()} {'='*20}")
