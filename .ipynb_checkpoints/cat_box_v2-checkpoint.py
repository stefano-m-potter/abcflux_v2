import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

def run_loso_analysis_for_target(target_col):
    """
    Performs a full Leave-One-Site-Out cross-validation for a given target variable,
    using a Box-Cox transformation on the target.
    """
    # 1. Load and prepare dataset
    file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_final.csv"
    df = pd.read_csv(file_path)
    df['land_cover'] = df['land_cover'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[df['flux_method'] == 'EC']

    # 2. Create tmean_C and date
    df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # 3. Define predictors and target
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

    # Drop rows with missing data for essential columns
    df = df.dropna(subset=['site_reference', target_col])

    # Define output paths
    out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv", target_col)
    os.makedirs(out_path, exist_ok=True)
    figures_path = os.path.join(out_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Prepare features (X) and target (y)
    X = df[feature_cols].copy()
    y = df[target_col]
    sites = df["site_reference"].unique()

    for col in categorical_features:
        X[col] = X[col].astype('category')

    results = []
    all_preds_df_list = []

    # Leave-One-Site-Out CV
    for test_site in sites:
        print(f"Processing site: {test_site} for target: {target_col}")
        train_idx = df["site_reference"] != test_site
        test_idx = df["site_reference"] == test_site

        if test_idx.sum() < 1:
            continue

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        dates_test = df.loc[test_idx, "date"]

        # --- Box-Cox Transformation ---
        # Shift data to be strictly positive for Box-Cox
        y_train_shifted = y_train.copy()
        shift = 0
        min_val = y_train_shifted.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6  # Add a small epsilon
            y_train_shifted += shift
        
        # Apply Box-Cox
        y_train_transformed, lmbda = boxcox(y_train_shifted)

        # Initialize and train the CatBoost model on transformed data
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
        model.fit(X_train, y_train_transformed)
        
        # Predict and inverse transform
        y_pred_transformed = model.predict(X_test)
        y_pred = inv_boxcox(y_pred_transformed, lmbda)
        
        # Reverse the shift if one was applied
        if shift > 0:
            y_pred -= shift

        site_df = pd.DataFrame({
            "Site": test_site, "Date": dates_test.values,
            "Observed": y_test.values, "Predicted": y_pred
        })
        all_preds_df_list.append(site_df)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Site": test_site, "RMSE": rmse, "MAE": mae, "R2": r2})

    # Combine results
    results_df = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

    # --- Save to disk with '_box' suffix ---
    results_csv_path = os.path.join(out_path, f'catboost_results_{target_col}_box.csv')
    predictions_csv_path = os.path.join(out_path, f'catboost_predictions_{target_col}_box.csv')
    results_df.to_csv(results_csv_path, index=False)
    all_preds_df.to_csv(predictions_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")
    print(f"Predictions saved to: {predictions_csv_path}")

    # Pooled metrics
    rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
    r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
    mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])
    
    print(f"\n--- Pooled Metrics for {target_col.upper()} ---")
    print(f"Pooled RMSE: {rmse_all:.4f}")
    print(f"Pooled MAE:  {mae_all:.4f}")
    print(f"Pooled R²:   {r2_all:.4f}")

    # Plotting
    unique_sites = all_preds_df["Site"].unique()
    if not unique_sites.any():
        print("\nNo sites to plot.")
    else:
        print("\nGenerating and saving individual site plots...")
        for site in unique_sites:
            fig, ax = plt.subplots(figsize=(12, 7))
            site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
            site_metrics = results_df[results_df["Site"] == site].iloc[0]
            
            ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
            ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
            ax.set_title(f"Observed vs. Predicted {target_col} (Box-Cox) for Site: {site}")
            ax.legend()
            ax.grid(True)
            fig.autofmt_xdate()
            
            textstr = f"RMSE: {site_metrics['RMSE']:.2f}\nMAE: {site_metrics['MAE']:.2f}\nR²: {site_metrics['R2']:.2f}"
            ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            # --- Save plot with '_box' suffix ---
            plot_filename = f'catboost_{target_col}_{site}_timeseries_box.png'
            plot_path = os.path.join(figures_path, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        print(f"All site plots saved to: {figures_path}")
        
    # --- SAVE THE FINAL MODEL ---
    final_model_path = '/explore/nobackup/people/spotter5/anna_v/v2/models'
    os.makedirs(final_model_path, exist_ok = True)
    
    # Final model training on all data with Box-Cox
    y_shifted = y.copy()
    shift = 0
    min_val = y_shifted.min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-6
        y_shifted += shift
    y_transformed, lmbda = boxcox(y_shifted)
    
    final_model = CatBoostRegressor(
        iterations=1200, learning_rate=0.01, depth=8, subsample=0.7,
        random_state=42, l2_leaf_reg=0.1, cat_features=categorical_features,
        verbose=0, allow_writing_files=False
    )
    
    # ✅ FIX: Convert lambda and shift to strings before saving in metadata
    metadata_to_save = {
        'lambda': str(lmbda), 
        'shift': str(shift)
    }
    final_model.set_params(metadata=metadata_to_save)
    
    # Fit the model
    final_model.fit(X, y_transformed)
    
    # Save the model, specifying json format to handle metadata well
    model_filename = f'{target_col}_box.json' # Use .json extension
    final_model_full_path = os.path.join(final_model_path, model_filename)
    final_model.save_model(final_model_full_path, format="json") 
    
    print(f"Final model for {target_col} saved to {final_model_full_path}")


if __name__ == '__main__':
    # List of target variables to run the analysis for
    targets_to_run = ['gpp', 'nee', 'reco', 'ch4_flux_total']
    
    for target in targets_to_run:
        print(f"\n{'='*20} RUNNING ANALYSIS FOR: {target.upper()} {'='*20}")
        run_loso_analysis_for_target(target_col=target)
        print(f"{'='*20} COMPLETED ANALYSIS FOR: {target.upper()} {'='*20}")