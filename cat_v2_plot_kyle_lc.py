#!/usr/bin/env python
# coding: utf-8

# ----- land cover will use Kyle's classification from CSV -----

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

def _coerce_numeric_if_possible(s: pd.Series) -> pd.Series:
    """
    Try to convert to numeric; if that creates many NaNs, fall back to original.
    """
    s_num = pd.to_numeric(s, errors='coerce')
    # If conversion produced mostly NaNs, keep original strings
    if s_num.isna().mean() > 0.5:
        return s.astype('string')
    return s_num

def run_loso_analysis(target_col):
    """
    Performs a full Leave-One-Site-Out (LOSO) cross-validation for a given target variable.
    Trains on all but one site, predicts the left-out site, aggregates results, saves per-site
    time series plots, pooled CSVs, a final model, and an Observed vs Predicted scatter plot.
    """
    # 1. Load and prepare dataset
    print(f"--- Processing Target: {target_col.upper()} ---")
    file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final.csv"
    df = pd.read_csv(file_path)

    # ---- JOIN: bring in Kyle's land cover classes ----
    lc_path = "/explore/nobackup/people/spotter5/anna_v/v2/all_landcover_classes_kaa.csv"
    lc = pd.read_csv(lc_path)
    lc.columns = lc.columns.str.strip()

    # Handle either spelling
    kyle_col = None
    for candidate in ["Reccomended by Kyle", "Reccomended by Kyle"]:
        if candidate in lc.columns:
            kyle_col = candidate
            break
    if kyle_col is None:
        raise KeyError("Could not find 'Reccomended By Kyle' (or 'Recommended By Kyle') in the landcover CSV.")

    if "site_reference" not in lc.columns:
        raise KeyError("Could not find 'site_reference' in the landcover CSV.")

    lc = lc[["site_reference", kyle_col]].dropna(subset=["site_reference"]).copy()
    # De-duplicate sites in the mapping, keeping the first occurrence
    lc = lc.drop_duplicates(subset=["site_reference"], keep="first")

    # Merge to main DF
    before = len(df)
    df = df.merge(lc, on="site_reference", how="left")
    unmatched = df[kyle_col].isna().sum()
    if unmatched > 0:
        print(f"⚠️  {unmatched} rows did not get a Kyle land-cover class (after joining on 'site_reference').")

    # Create 'land_cover' from Kyle's column
    # Try numeric; if mostly non-numeric, keep as strings (CatBoost can handle categoricals)
    lc_series = _coerce_numeric_if_possible(df[kyle_col])
    df["land_cover"] = lc_series
    # If numeric, cast to Int64 (nullable); otherwise to string category
    if pd.api.types.is_numeric_dtype(df["land_cover"]):
        df["land_cover"] = df["land_cover"].astype("Int64")
    else:
        df["land_cover"] = df["land_cover"].astype("string")

    # Keep month numeric but we will mark it categorical in X later
    df["month"] = df["month"].astype(int)
    # Filter to EC flux only
    df = df[df["flux_method"] == "EC"]

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
        'Percent_NonVegetated', 'Percent_Tree_Cover', 'sm_surface', 'sm_rootzone',
        'snow_cover', 'snow_depth'
    ]
    categorical_features = ['land_cover', 'month']

    # Drop rows where the current target or site_reference is missing
    df = df.dropna(subset=['site_reference', target_col])

    # 4. Define output paths
    loocv_out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv_kyle", target_col)
    figures_path = os.path.join(loocv_out_path, "figures")
    models_out_path = "/explore/nobackup/people/spotter5/anna_v/v2/models_kyle"
    os.makedirs(loocv_out_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_out_path, exist_ok=True)

    # 5. Prepare features (X) and target (y)
    # Ensure 'land_cover' exists
    if "land_cover" not in df.columns:
        raise KeyError("land_cover column missing after merge. Check source CSV and join keys.")
    X = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    # Mark categoricals (CatBoost handles category/string types)
    for col in categorical_features:
        if col == "land_cover":
            # If numeric nullable, keep as category; if string-like, also category
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype("category")

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

        model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.01,
            depth=8,
            subsample=0.7,
            random_state=42,
            l2_leaf_reg=0.1,
            rsm=0.8,
            cat_features=categorical_features,  # names OK with pandas Pool
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

    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    print(f"\n  --- Pooled Metrics for {target_col.upper()} ---")
    print(f"  Pooled R²: {r2_all:.4f}, Pooled RMSE: {rmse_all:.4f}, Pooled MAE: {mae_all:.4f}")

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

    # 8. Per-site plots
    print("\n  Generating and saving individual site plots...")
    for site in all_preds_df["Site"].unique():
        fig, ax = plt.subplots(figsize=(12, 7))
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()

        textstr = f"RMSE: {site_metrics['RMSE']:.2f}\nMAE: {site_metrics['MAE']:.2f}\nR²: {site_metrics['R2']:.2f}"
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        plot_filename = f'catboost_{target_col}_{site}_timeseries_cat.png'
        plot_path = os.path.join(figures_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  All site plots saved to: {figures_path}")

    # 8b. Pooled density plot
    pooled = all_preds_df[['Observed', 'Predicted']].replace([np.inf, -np.inf], np.nan).dropna()
    if not pooled.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        lo = np.nanmin([pooled['Observed'].min(), pooled['Predicted'].min()])
        hi = np.nanmax([pooled['Observed'].max(), pooled['Predicted'].max()])
        pad = 0.05 * (hi - lo if np.isfinite(hi - lo) and (hi - lo) > 0 else 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        hb = ax.hexbin(pooled['Observed'], pooled['Predicted'], gridsize=80, cmap='Reds', bins='log', mincnt=1)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('log10(N points)')
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='black', linewidth=1.5)
        ax.set_title(f'Observed vs Predicted (LOSO) — {target_col}')
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.grid(True)
        annot = f"R² = {r2_all:.2f}\nRMSE = {rmse_all:.2f}\nMAE = {mae_all:.2f}"
        ax.text(0.97, 0.03, annot, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        pooled_plot_path = os.path.join(figures_path, f'catboost_{target_col}_obs_vs_pred_all_sites_cat.png')
        plt.savefig(pooled_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Pooled Observed vs Predicted plot saved to: {pooled_plot_path}")
    else:
        print("  Skipped pooled density plot: no valid Observed/Predicted pairs after cleaning.")

    # 9. Train and Save Final Model on all data
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
    targets_to_run = ['gpp', 'nee', 'reco', 'ch4_flux_total']
    for target in targets_to_run:
        print(f"\n{'='*50}\nRUNNING ANALYSIS FOR: {target.upper()}\n{'='*50}")
        run_loso_analysis(target_col=target)
        print(f"\n{'='*50}\nCOMPLETED ANALYSIS FOR: {target.upper()}\n{'='*50}")
