#!/usr/bin/env python
# coding: utf-8

# ----- land cover will use esa_cci classification from all_landcover_classes_v2.csv -----

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

def _coerce_numeric_if_possible(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors='coerce')
    return s_num if s_num.notna().mean() >= 0.5 else s.astype('string')

def run_loso_analysis(target_col):
    print(f"--- Processing Target: {target_col.upper()} ---")
    df = pd.read_csv("/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final.csv")

    # ---- JOIN: bring in esa_cci land cover ----
    lc_path = "/explore/nobackup/people/spotter5/anna_v/v2/all_landcover_classes_v2.csv"
    lc = pd.read_csv(lc_path)
    lc.columns = lc.columns.str.strip()

    if "site_reference" not in lc.columns or "esa_cci" not in lc.columns:
        raise KeyError("Expected columns 'site_reference' and 'esa_cci' in all_landcover_classes_v2.csv")

    lc = lc[["site_reference", "esa_cci"]].drop_duplicates(subset=["site_reference"])
    df = df.merge(lc, on="site_reference", how="left")
    if df["esa_cci"].isna().any():
        print(f"⚠️  {df['esa_cci'].isna().sum()} rows have no esa_cci classification.")

    # Create land_cover from esa_cci
    lc_series = _coerce_numeric_if_possible(df["esa_cci"])
    df["land_cover"] = lc_series.astype("Int64") if pd.api.types.is_numeric_dtype(lc_series) else lc_series.astype("string")

    df["month"] = df["month"].astype(int)
    df = df[df["flux_method"] == "EC"]

    # Derived features
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # Features and categorical fields
    feature_cols = [
        'EVI','NDVI','sur_refl_b01','sur_refl_b02','sur_refl_b03',
        'sur_refl_b07','NDWI','pdsi','srad','tmean_C','vap','vs',
        'bdod_0_100cm','cec_0_100cm','cfvo_0_100cm','clay_0_100cm',
        'nitrogen_0_100cm','ocd_0_100cm','phh2o_0_100cm','sand_0_100cm',
        'silt_0_100cm','soc_0_100cm','co2_cont','ALT',
        'land_cover','month',
        'lai','fpar','Percent_NonTree_Vegetation','Percent_NonVegetated',
        'Percent_Tree_Cover','sm_surface','sm_rootzone','snow_cover','snow_depth'
    ]
    categorical_features = ['land_cover','month']

    df = df.dropna(subset=['site_reference', target_col])

    # Output paths
    loocv_out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv_kyle", target_col)
    figures_path = os.path.join(loocv_out_path, "figures")
    models_out_path = "/explore/nobackup/people/spotter5/anna_v/v2/models_kyle"
    os.makedirs(loocv_out_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_out_path, exist_ok=True)

    X = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    for col in categorical_features:
        X[col] = X[col].astype("category")

    results, all_preds_df_list = [], []

    # LOSO CV
    for test_site in sites:
        print(f"  Processing site: {test_site}...")
        train_idx = df["site_reference"] != test_site
        test_idx  = df["site_reference"] == test_site
        if test_idx.sum() < 1: continue

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test,  y_test  = X.loc[test_idx], y.loc[test_idx]
        dates_test = df.loc[test_idx, "date"]

        model = CatBoostRegressor(
            iterations=1200, learning_rate=0.01, depth=8, subsample=0.7,
            random_state=42, l2_leaf_reg=0.1, rsm=0.8,
            cat_features=categorical_features, verbose=0, allow_writing_files=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        site_df = pd.DataFrame({"Site": test_site, "Date": dates_test,
                                "Observed": y_test, "Predicted": y_pred})
        all_preds_df_list.append(site_df)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        results.append({"Site": test_site, "RMSE": rmse, "MAE": mae, "R2": r2})

    if not results:
        print(f"No data processed for target '{target_col}'. Skipping.")
        return

    # Save metrics and predictions
    results_df = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)
    results_df.to_csv(os.path.join(loocv_out_path, f'catboost_results_{target_col}_cat.csv'), index=False)
    all_preds_df.to_csv(os.path.join(loocv_out_path, f'catboost_predictions_{target_col}_cat.csv'), index=False)

    # Report pooled metrics
    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all   = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all  = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    print(f"Pooled R²: {r2_all:.4f}, RMSE: {rmse_all:.4f}, MAE: {mae_all:.4f}")

    # Train and save final model
    final_model = CatBoostRegressor(
        iterations=1200, learning_rate=0.01, depth=8, subsample=0.7,
        random_state=42, l2_leaf_reg=0.1, rsm=0.8,
        cat_features=categorical_features, verbose=0, allow_writing_files=False
    )
    final_model.fit(X, y)
    final_model.save_model(os.path.join(models_out_path, f'{target_col}.json'))
    print(f"Final model saved for {target_col}.")

if __name__ == '__main__':
    for target in ['gpp','nee','reco','ch4_flux_total']:
        print(f"\n{'='*50}\nRUNNING ANALYSIS FOR: {target.upper()}\n{'='*50}")
        run_loso_analysis(target)
        print(f"{'='*50}\nCOMPLETED ANALYSIS FOR: {target.upper()}\n{'='*50}")
