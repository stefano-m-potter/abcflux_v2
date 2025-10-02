#!/usr/bin/env python
# coding: utf-8

import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_CSV         = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final.csv"
TOWERS_CSV       = "/explore/nobackup/people/spotter5/anna_v/v2/methane_towers.csv"
OUT_BASE_LOOCV   = "/explore/nobackup/people/spotter5/anna_v/v2/loocv_methane"
OUT_BASE_MODELS  = "/explore/nobackup/people/spotter5/anna_v/v2/models_methane"

def safe_slug(text: str, maxlen: int = 120) -> str:
    """Make a filesystem-safe filename fragment from arbitrary text."""
    s = str(text).strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:maxlen] if s else "site"

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² that returns NaN if y_true has zero variance or too few samples."""
    y_true = np.asarray(y_true)
    if y_true.size < 2 or np.nanstd(y_true) == 0:
        return np.nan
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan

def run_loso_analysis(target_col: str):
    """
    LOSO CV for a single target (ch4_flux_total).
    - Filters expert_flag_ch4 != 4 (if column present)
    - **Cold rule**: after deriving tmean_C, sets sm_surface/sm_rootzone = 0 where tmean_C < -10
    - Training rows are restricted to sites listed in methane_towers.csv
      (test site is always the left-out site, even if not in the towers list)
    - Saves per-site plots, pooled metrics, pooled obs-vs-pred density, and final model
    """
    print(f"--- Processing Target: {target_col.upper()} ---")

    # ---- 1) Load data ----
    df = pd.read_csv(DATA_CSV)

    # Ensure required columns / types
    if 'site_reference' not in df.columns:
        raise ValueError("Input dataframe must contain 'site_reference'.")
    df['site_reference'] = df['site_reference'].astype(str)

    if 'land_cover' in df.columns:
        df['land_cover'] = df['land_cover'].astype(int)
    if 'month' in df.columns:
        df['month'] = df['month'].astype(int)

    # Filter to EC sites if column exists
    if 'flux_method' in df.columns:
        df = df[df['flux_method'] == 'EC']

    # Filter out low-quality methane flags if column exists
    if 'expert_flag_ch4' in df.columns:
        before_n = len(df)
        df = df[df['expert_flag_ch4'] != 4]
        print(f"Filtered expert_flag_ch4 == 4: {before_n - len(df)} rows removed (remaining {len(df)}).")
    else:
        print("Note: 'expert_flag_ch4' not found; no expert-flag filter applied.")

    # Require target column and site_reference
    df = df.dropna(subset=['site_reference', target_col])

    # ---- 2) Derived features ----
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1), errors='coerce')

    # **COLD RULE**: zero-out soil moisture predictors when tmean_C < -10
    if 'sm_surface' in df.columns and 'sm_rootzone' in df.columns:
        mask = df['tmean_C'] < -10
        df.loc[mask, 'sm_surface'] = 0.0
        df.loc[mask, 'sm_rootzone'] = 0.0
        print(f"Applied cold-climate rule: set sm_surface/sm_rootzone=0 for {mask.sum()} rows (tmean_C < -10).")
    else:
        print("Warning: sm_surface and/or sm_rootzone columns not found; cold-climate rule not applied.")

    # ---- 3) Predictors / categorical ----
    feature_cols = [
        'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03',
        'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
        'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm', 'swe',
        'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
        'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
        'land_cover', 'month',
        'lai', 'fpar', 'Percent_NonTree_Vegetation',
        'Percent_NonVegetated', 'Percent_Tree_Cover',
        'sm_surface', 'sm_rootzone',
        'snow_cover', 'snow_depth'
    ]
    categorical_features = ['land_cover', 'month']

    # Keep only columns that exist
    existing_cols = [c for c in feature_cols if c in df.columns]
    missing = sorted(set(feature_cols) - set(existing_cols))
    if missing:
        print(f"Warning: missing predictor columns dropped: {missing}")

    X_full = df[existing_cols].copy()
    y_full = df[target_col].copy()

    # Cast categoricals if present
    for col in (set(categorical_features) & set(existing_cols)):
        X_full[col] = X_full[col].astype('category')

    sites_all = df["site_reference"].unique()

    # ---- 4) Towers allow-list for TRAINING ONLY ----
    towers_df = pd.read_csv(TOWERS_CSV)
    if 'site_reference' not in towers_df.columns:
        raise ValueError("methane_towers.csv must contain a 'site_reference' column.")
    allowed_train_sites = set(towers_df['site_reference'].astype(str).unique())
    print(f"Training will be restricted to {len(allowed_train_sites)} methane tower sites.")

    # ---- 5) Output dirs ----
    loocv_out_path = os.path.join(OUT_BASE_LOOCV, target_col)
    figures_path = os.path.join(loocv_out_path, "figures")
    models_out_path = OUT_BASE_MODELS
    os.makedirs(loocv_out_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_out_path, exist_ok=True)

    results = []
    all_preds_df_list = []

    # ---- 6) LOSO loop ----
    for test_site in sites_all:
        print(f"  Processing site: {test_site}...")

        test_mask = (df["site_reference"] == test_site)
        if test_mask.sum() < 1:
            print(f"    Skipping {test_site}: no test rows.")
            continue

        # Train on all other sites but restricted to allowed methane towers
        train_mask = (df["site_reference"] != test_site) & (df["site_reference"].isin(allowed_train_sites))
        if train_mask.sum() < 1:
            print(f"    Skipping {test_site}: empty training set after tower restriction.")
            continue

        X_train, y_train = X_full.loc[train_mask], y_full.loc[train_mask]
        X_test,  y_test  = X_full.loc[test_mask],  y_full.loc[test_mask]
        dates_test = df.loc[test_mask, "date"]

        model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.01,
            depth=8,
            subsample=0.7,
            random_state=42,
            l2_leaf_reg=0.1,
            rsm=0.8,
            cat_features=[c for c in categorical_features if c in existing_cols],
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
        r2   = safe_r2(y_test, y_pred)   # robust R²
        mae  = mean_absolute_error(y_test, y_pred)
        results.append({"Site": test_site, "RMSE": rmse, "MAE": mae, "R2": r2})

    if not results:
        print(f"No folds completed for '{target_col}'. Check filters / allow-list.")
        return

    # ---- 7) Save results / pooled metrics ----
    results_df = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

    results_csv_path = os.path.join(loocv_out_path, f'catboost_results_{target_col}_cat.csv')
    predictions_csv_path = os.path.join(loocv_out_path, f'catboost_predictions_{target_col}_cat.csv')
    results_df.to_csv(results_csv_path, index=False)
    all_preds_df.to_csv(predictions_csv_path, index=False)
    print(f"\n  Results saved to: {results_csv_path}")

    # Pooled metrics (guard against zero-variance pooled y)
    pooled = all_preds_df[['Observed', 'Predicted']].replace([np.inf, -np.inf], np.nan).dropna()
    if not pooled.empty and np.nanstd(pooled['Observed']) > 0:
        rmse_all = np.sqrt(mean_squared_error(pooled["Observed"], pooled["Predicted"]))
        r2_all   = r2_score(pooled["Observed"], pooled["Predicted"])
        mae_all  = mean_absolute_error(pooled["Observed"], pooled["Predicted"])
    else:
        rmse_all = r2_all = mae_all = np.nan

    print(f"\n  --- Pooled Metrics for {target_col.upper()} ---")
    print(f"  Pooled R²: {r2_all:.4f}, Pooled RMSE: {rmse_all:.4f}, Pooled MAE: {mae_all:.4f}")

    # Summary across sites (ignore NaN R²)
    mean_r2     = float(np.nanmean(results_df['R2'])) if results_df['R2'].notna().any() else np.nan
    median_r2   = float(np.nanmedian(results_df['R2'])) if results_df['R2'].notna().any() else np.nan
    mean_rmse   = results_df['RMSE'].mean()
    median_rmse = results_df['RMSE'].median()
    mean_mae    = results_df['MAE'].mean()
    median_mae  = results_df['MAE'].median()

    print(f"\n  --- Summary Metrics Across Sites for {target_col.upper()} ---")
    print(f"  Mean R²:   {mean_r2:.4f}, Median R²:   {median_r2:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f}, Median RMSE: {median_rmse:.4f}")
    print(f"  Mean MAE:  {mean_mae:.4f}, Median MAE:  {median_mae:.4f}")

    # ---- 8) Per-site plots (safe filenames) ----
    print("\n  Generating and saving individual site plots...")
    for site in all_preds_df["Site"].unique():
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()

        r2_text = site_metrics['R2'] if pd.notna(site_metrics['R2']) else float('nan')
        textstr = f"RMSE: {site_metrics['RMSE']:.2f}\nMAE: {site_metrics['MAE']:.2f}\nR²: {r2_text:.2f}"
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        site_slug = safe_slug(site)
        plot_path = os.path.join(figures_path, f'catboost_{target_col}_{site_slug}_timeseries_cat.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  All site plots saved to: {figures_path}")

    # ---- 8b) Pooled density plot ----
    if not pooled.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        lo = float(np.nanmin([pooled['Observed'].min(), pooled['Predicted'].min()]))
        hi = float(np.nanmax([pooled['Observed'].max(), pooled['Predicted'].max()]))
        span = hi - lo if np.isfinite(hi - lo) and (hi - lo) > 0 else 1.0
        pad = 0.05 * span
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        hb = ax.hexbin(
            pooled['Observed'], pooled['Predicted'],
            gridsize=80, cmap='Reds', bins='log', mincnt=1
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('log10(N points)')

        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='black', linewidth=1.5)
        ax.set_title(f'Observed vs Predicted (LOSO) — {target_col}')
        ax.set_xlabel('Observed'); ax.set_ylabel('Predicted'); ax.grid(True)

        annot = f"R² = {r2_all:.2f}\nRMSE = {rmse_all:.2f}\nMAE = {mae_all:.2f}"
        ax.text(
            0.97, 0.03, annot, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

        pooled_plot_path = os.path.join(figures_path, f'catboost_{target_col}_obs_vs_pred_all_sites_cat.png')
        plt.savefig(pooled_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Pooled Observed vs Predicted plot saved to: {pooled_plot_path}")
    else:
        print("  Skipped pooled density plot: no valid Observed/Predicted pairs after cleaning.")

    # ---- 9) Final model trained on ALL DATA *restricted to allowed training towers* ----
    print("\n  Training and saving final model on all training-allowed towers...")
    final_mask = df["site_reference"].isin(allowed_train_sites)
    if final_mask.sum() < 1:
        print("  No rows available after tower restriction for final model. Skipping final model save.")
        return

    X_final = X_full.loc[final_mask].copy()
    y_final = y_full.loc[final_mask].copy()
    final_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.01,
        depth=8,
        subsample=0.7,
        random_state=42,
        l2_leaf_reg=0.1,
        rsm=0.8,
        cat_features=[c for c in categorical_features if c in existing_cols],
        verbose=0,
        allow_writing_files=False
    )
    final_model.fit(X_final, y_final)
    os.makedirs(OUT_BASE_MODELS, exist_ok=True)
    model_path = os.path.join(OUT_BASE_MODELS, f'{target_col}.json')
    final_model.save_model(model_path)
    print(f"  Final model saved to: {model_path}")

if __name__ == '__main__':
    # Only target ch4_flux_total
    run_loso_analysis(target_col='ch4_flux_total')
