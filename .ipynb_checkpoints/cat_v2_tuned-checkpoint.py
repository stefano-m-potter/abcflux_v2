#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

def run_loso_analysis(target_col):
    """
    Performs a full Leave-One-Site-Out (LOSO) cross-validation for a given target variable
    using CatBoost with nested cross-validation for hyperparameter tuning on a CPU.

    Args:
        target_col (str): The name of the target variable column (e.g., 'gpp', 'nee').
    """
    # 1. Load and prepare dataset
    print(f"--- Processing Target: {target_col.upper()} ---")
    file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final.csv"
    df = pd.read_csv(file_path)
    df['land_cover'] = df['land_cover'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[df['flux_method'] == 'EC']

    # 2. Create derived features
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # 3. Define predictors
    feature_cols = [
        'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03',
        'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
        'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
        'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
        'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
        'land_cover', 'month',
        'lai', 'fpar', 'Percent_NonTree_Vegetation',
         'Percent_NonVegetated', 'Percent_Tree_Cover',  'sm_surface', 'sm_rootzone',
        'snow_cover',
        'snow_depth', 'NDSI_snow_cover'
    ]
    categorical_features_names = ['land_cover', 'month']
    
    # Drop rows where the current target or site_reference is missing
    df = df.dropna(subset=['site_reference', target_col])

    # 4. Define output paths
    loocv_out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv", target_col)
    figures_path = os.path.join(loocv_out_path, "figures_catboost_tuned")
    models_out_path = '/explore/nobackup/people/spotter5/anna_v/v2/models'
    os.makedirs(loocv_out_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_out_path, exist_ok=True)

    # 5. Prepare features (X) and target (y)
    X = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    results = []
    all_preds_df_list = []

    # 6. Nested CV: Outer loop is LOSO, Inner loop is GridSearchCV for tuning
    for test_site in sites:
        print(f"  Processing site: {test_site}...")
        train_idx = df["site_reference"] != test_site
        test_idx = df["site_reference"] == test_site

        if test_idx.sum() < 1:
            continue

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        dates_test = df.loc[test_idx, "date"]

        # --- A. DEFINE THE MODEL AND PARAMETER GRID FOR TUNING ---
        model = CatBoostRegressor(
            random_state=42,
            verbose=0,  # GridSearchCV will have its own verbosity
            allow_writing_files=False,
            cat_features=categorical_features_names
        )
        
        param_grid = {
            'iterations': [1200],
            'depth': [5, 8, 12],
            'learning_rate': [0.01],
            'subsample': [0.7, 0.9],
            'l2_leaf_reg': [0.01, 1, 5] # L2 regularization
        }

        # --- B. SETUP AND RUN THE INNER CROSS-VALIDATION (GRID SEARCH) ---
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1 # Shows progress
        )
        grid_search.fit(X_train, y_train)

        # --- C. EVALUATE ON THE OUTER TEST SET ---
        print(f"    Best params for this fold: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        site_df = pd.DataFrame({"Site": test_site, "Date": dates_test.values, "Observed": y_test.values, "Predicted": y_pred})
        all_preds_df_list.append(site_df)
        
        results.append({
            "Site": test_site,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "Best_Params": str(grid_search.best_params_)
        })

    # 7. Combine, Save, and Report Results
    results_df = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

    results_csv_path = os.path.join(loocv_out_path, f'catboost_tuned_results_{target_col}.csv')
    predictions_csv_path = os.path.join(loocv_out_path, f'catboost_tuned_predictions_{target_col}.csv')
    results_df.to_csv(results_csv_path, index=False)
    all_preds_df.to_csv(predictions_csv_path, index=False)
    print(f"\n  Tuned CatBoost results saved to: {results_csv_path}")

    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    print(f"\n  --- Pooled Metrics for Tuned CatBoost: {target_col.upper()} ---")
    print(f"  Pooled R²: {r2_all:.4f}, Pooled RMSE: {rmse_all:.4f}, Pooled MAE: {mae_all:.4f}")

    # 8. Plotting (Code omitted for brevity, but it would go here)

    # 9. Find Best Params on ALL Data and Save Final Model
    print("\n  Finding best params and training final model on all data...")
    final_model_base = CatBoostRegressor(
        random_state=42,
        verbose=0,
        allow_writing_files=False,
        cat_features=categorical_features_names
    )
    final_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    final_grid_search = GridSearchCV(estimator=final_model_base, param_grid=param_grid, cv=final_cv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    final_grid_search.fit(X, y)
    
    print(f"  --- Best params for final model: {final_grid_search.best_params_} ---")
    final_model = final_grid_search.best_estimator_
    
    model_filename = os.path.join(models_out_path, f'catboost_tuned_{target_col}.json')
    final_model.save_model(model_filename)
    print(f"  Final tuned CatBoost model saved to: {model_filename}")


if __name__ == '__main__':
    targets_to_run = ['nee', 'gpp', 'reco', 'ch4_flux_total']

    for target in targets_to_run:
        print(f"\n{'='*50}\nRUNNING TUNED CATBOOST (GPU) ANALYSIS FOR: {target.upper()}\n{'='*50}")
        run_loso_analysis(target_col=target)
        print(f"\n{'='*50}\nCOMPLETED TUNED CATBOOST (GPU) ANALYSIS FOR: {target.upper()}\n{'='*50}")