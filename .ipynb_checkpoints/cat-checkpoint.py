#!/usr/bin/env python
# coding: utf-8

# 10-fold gpp

# In[7]:


import os
import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
predictor_vars = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features
]
target_var = 'gpp'
categorical_features = ['land_cover', 'month']

# 4. Drop rows only if the target variable is missing
# CatBoost can also handle missing values (NaN) in predictor variables.
df_model = df.dropna(subset=[target_var]).copy()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    df_model[col] = df_model[col].astype('category')

# 5. Define features (X) and target (y)
X = df_model[predictor_vars]
y = df_model[target_var]

# 6. Define parameter grid for CatBoost
# Note: CatBoost uses 'depth' instead of 'max_depth', 'iterations' for 'n_estimators'
param_grid = {
    'depth': [4, 6, 8, 10, 12, 14, 16],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [0.1, 0.5, 1, 3, 5],
    'iterations': [700, 1000, 1500]
}

# 7. Model and 10-fold CV
# Pass the list of categorical features directly to the model
cat_model = CatBoostRegressor(
    random_state=42,
    cat_features=categorical_features,
    verbose=0, # Suppress verbose output during training
    allow_writing_files=False # Suppress creation of catboost_info dir
)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 8. Grid search (optimize RMSE)
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X, y)

# 9. Metrics
best_rmse = -grid_search.best_score_
best_params = grid_search.best_params_
print(f"✅ Best RMSE: {best_rmse:.3f}")
print("✅ Best Parameters:", best_params)

# 10. R² using the best estimator from grid search
r2_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# 11. MAE using the best estimator from grid search
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mae_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring=mae_scorer,
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV MAE: {-np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")


# In[8]:


't'


# LOOSO gpp

# In[9]:


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

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
feature_cols = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features are now included directly
]
target_col = 'gpp'
categorical_features = ['land_cover', 'month']

# Drop rows only if the target variable or site_reference is missing.
# CatBoost will handle missing values in the numerical predictor variables.
df = df.dropna(subset=['site_reference', target_col])

# Define output path for CSVs and create it
out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv", target_col)
os.makedirs(out_path, exist_ok=True)

# Define a separate output path for figures and create it
figures_path = os.path.join(out_path, "figures")
os.makedirs(figures_path, exist_ok=True)


# Prepare features (X) and target (y)
# No one-hot encoding is needed for CatBoost
X = df[feature_cols]
y = df[target_col]
sites = df["site_reference"].unique()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    X[col] = X[col].astype('category')

results = []
all_preds_df_list = []

# Leave-One-Site-Out CV
for test_site in sites:
    print(f"Processing site: {test_site}...")
    train_idx = df["site_reference"] != test_site
    test_idx = df["site_reference"] == test_site

    if test_idx.sum() < 1:
        continue

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    dates_test = df.loc[test_idx, "date"]

    # Initialize and train the CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        subsample=0.7,
        l2_leaf_reg = 1,
        random_state=42,
        cat_features=categorical_features,
        verbose=0, # Suppress verbose output
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

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Site": test_site,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

# Combine results
results_df = pd.DataFrame(results)
all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

# Save to disk with '_cat' suffix
results_csv_path = os.path.join(out_path, f'catboost_results_{target_col}_cat.csv')
predictions_csv_path = os.path.join(out_path, f'catboost_predictions_{target_col}_cat.csv')
results_df.to_csv(results_csv_path, index=False)
all_preds_df.to_csv(predictions_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")
print(f"Predictions saved to: {predictions_csv_path}")


# Pooled metrics
rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])

print("\n--- Site-Specific Results ---")
print(results_df)
print("\n--- Pooled Metrics ---")
print(f"Pooled RMSE: {rmse_all:.4f}")
print(f"Pooled MAE:  {mae_all:.4f}")
print(f"Pooled R²:   {r2_all:.4f}")

# Median metrics across sites
median_rmse = results_df["RMSE"].median()
median_mae = results_df["MAE"].median()
median_r2 = results_df["R2"].median()

print("\n--- Median Metrics Across Sites ---")
print(f"Median RMSE: {median_rmse:.4f}")
print(f"Median MAE:  {median_mae:.4f}")
print(f"Median R²:   {median_r2:.4f}")

# --- Plotting ---
# Loop through each site and save a separate plot
unique_sites = all_preds_df["Site"].unique()
if not unique_sites.any():
    print("\nNo sites to plot.")
else:
    print("\nGenerating and saving individual site plots...")
    for site in unique_sites:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]
        rmse_val = round(site_metrics["RMSE"], 2)
        r2_val = round(site_metrics["R2"], 2)
        mae_val = round(site_metrics["MAE"], 2)

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate() # Auto-formats the x-axis labels for dates

        # Add metrics text to the plot
        textstr = f"RMSE: {rmse_val}\nMAE: {mae_val}\nR²: {r2_val}"
        ax.text(
            0.97, 0.03, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        
        # Define the output path for the plot
        plot_filename = f'catboost_{target_col}_{site}_timeseries_cat.png'
        plot_path = os.path.join(figures_path, plot_filename)
        
        # Save the figure
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Close the plot to free up memory
        plt.close(fig)
        
    print(f"All site plots saved to: {figures_path}")


# 10-fold nee

# In[10]:


import os
import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
predictor_vars = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features
]
target_var = 'nee'
categorical_features = ['land_cover', 'month']

# 4. Drop rows only if the target variable is missing
# CatBoost can also handle missing values (NaN) in predictor variables.
df_model = df.dropna(subset=[target_var]).copy()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    df_model[col] = df_model[col].astype('category')

# 5. Define features (X) and target (y)
X = df_model[predictor_vars]
y = df_model[target_var]

# 6. Define parameter grid for CatBoost
# Note: CatBoost uses 'depth' instead of 'max_depth', 'iterations' for 'n_estimators'
param_grid = {
    'depth': [4, 6, 8, 10, 12, 14, 16],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [0.1, 0.5, 1, 3, 5],
    'iterations': [700, 1000, 1500]
}

# 7. Model and 10-fold CV
# Pass the list of categorical features directly to the model
cat_model = CatBoostRegressor(
    random_state=42,
    cat_features=categorical_features,
    verbose=0, # Suppress verbose output during training
    allow_writing_files=False # Suppress creation of catboost_info dir
)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 8. Grid search (optimize RMSE)
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X, y)

# 9. Metrics
best_rmse = -grid_search.best_score_
best_params = grid_search.best_params_
print(f"✅ Best RMSE: {best_rmse:.3f}")
print("✅ Best Parameters:", best_params)

# 10. R² using the best estimator from grid search
r2_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# 11. MAE using the best estimator from grid search
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mae_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring=mae_scorer,
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV MAE: {-np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")


# LOOSO NEE

# In[ ]:


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

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
feature_cols = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features are now included directly
]
target_col = 'nee'
categorical_features = ['land_cover', 'month']

# Drop rows only if the target variable or site_reference is missing.
# CatBoost will handle missing values in the numerical predictor variables.
df = df.dropna(subset=['site_reference', target_col])

# Define output path for CSVs and create it
out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv", target_col)
os.makedirs(out_path, exist_ok=True)

# Define a separate output path for figures and create it
figures_path = os.path.join(out_path, "figures")
os.makedirs(figures_path, exist_ok=True)


# Prepare features (X) and target (y)
# No one-hot encoding is needed for CatBoost
X = df[feature_cols]
y = df[target_col]
sites = df["site_reference"].unique()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    X[col] = X[col].astype('category')

results = []
all_preds_df_list = []

# Leave-One-Site-Out CV
for test_site in sites:
    print(f"Processing site: {test_site}...")
    train_idx = df["site_reference"] != test_site
    test_idx = df["site_reference"] == test_site

    if test_idx.sum() < 1:
        continue

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    dates_test = df.loc[test_idx, "date"]

    # Initialize and train the CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        subsample=0.7,
        l2_leaf_reg = 1,
        random_state=42,
        cat_features=categorical_features,
        verbose=0, # Suppress verbose output
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

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Site": test_site,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

# Combine results
results_df = pd.DataFrame(results)
all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

# Save to disk with '_cat' suffix
results_csv_path = os.path.join(out_path, f'catboost_results_{target_col}_cat.csv')
predictions_csv_path = os.path.join(out_path, f'catboost_predictions_{target_col}_cat.csv')
results_df.to_csv(results_csv_path, index=False)
all_preds_df.to_csv(predictions_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")
print(f"Predictions saved to: {predictions_csv_path}")


# Pooled metrics
rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])

print("\n--- Site-Specific Results ---")
print(results_df)
print("\n--- Pooled Metrics ---")
print(f"Pooled RMSE: {rmse_all:.4f}")
print(f"Pooled MAE:  {mae_all:.4f}")
print(f"Pooled R²:   {r2_all:.4f}")

# Median metrics across sites
median_rmse = results_df["RMSE"].median()
median_mae = results_df["MAE"].median()
median_r2 = results_df["R2"].median()

print("\n--- Median Metrics Across Sites ---")
print(f"Median RMSE: {median_rmse:.4f}")
print(f"Median MAE:  {median_mae:.4f}")
print(f"Median R²:   {median_r2:.4f}")

# --- Plotting ---
# Loop through each site and save a separate plot
unique_sites = all_preds_df["Site"].unique()
if not unique_sites.any():
    print("\nNo sites to plot.")
else:
    print("\nGenerating and saving individual site plots...")
    for site in unique_sites:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]
        rmse_val = round(site_metrics["RMSE"], 2)
        r2_val = round(site_metrics["R2"], 2)
        mae_val = round(site_metrics["MAE"], 2)

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate() # Auto-formats the x-axis labels for dates

        # Add metrics text to the plot
        textstr = f"RMSE: {rmse_val}\nMAE: {mae_val}\nR²: {r2_val}"
        ax.text(
            0.97, 0.03, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        
        # Define the output path for the plot
        plot_filename = f'catboost_{target_col}_{site}_timeseries_cat.png'
        plot_path = os.path.join(figures_path, plot_filename)
        
        # Save the figure
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Close the plot to free up memory
        plt.close(fig)
        
    print(f"All site plots saved to: {figures_path}")


# 10 fold reco

# In[ ]:


import os
import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
predictor_vars = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features
]
target_var = 'reco'
categorical_features = ['land_cover', 'month']

# 4. Drop rows only if the target variable is missing
# CatBoost can also handle missing values (NaN) in predictor variables.
df_model = df.dropna(subset=[target_var]).copy()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    df_model[col] = df_model[col].astype('category')

# 5. Define features (X) and target (y)
X = df_model[predictor_vars]
y = df_model[target_var]

# 6. Define parameter grid for CatBoost
# Note: CatBoost uses 'depth' instead of 'max_depth', 'iterations' for 'n_estimators'
param_grid = {
    'depth': [4, 6, 8, 10, 12, 14, 16],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [0.1, 0.5, 1, 3, 5],
    'iterations': [700, 1000, 1500]
}

# 7. Model and 10-fold CV
# Pass the list of categorical features directly to the model
cat_model = CatBoostRegressor(
    random_state=42,
    cat_features=categorical_features,
    verbose=0, # Suppress verbose output during training
    allow_writing_files=False # Suppress creation of catboost_info dir
)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 8. Grid search (optimize RMSE)
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X, y)

# 9. Metrics
best_rmse = -grid_search.best_score_
best_params = grid_search.best_params_
print(f"✅ Best RMSE: {best_rmse:.3f}")
print("✅ Best Parameters:", best_params)

# 10. R² using the best estimator from grid search
r2_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# 11. MAE using the best estimator from grid search
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mae_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring=mae_scorer,
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV MAE: {-np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")


# In[ ]:


#LOOSO RECO


# In[ ]:


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

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
feature_cols = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features are now included directly
]
target_col = 'reco'
categorical_features = ['land_cover', 'month']

# Drop rows only if the target variable or site_reference is missing.
# CatBoost will handle missing values in the numerical predictor variables.
df = df.dropna(subset=['site_reference', target_col])

# Define output path for CSVs and create it
out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv", target_col)
os.makedirs(out_path, exist_ok=True)

# Define a separate output path for figures and create it
figures_path = os.path.join(out_path, "figures")
os.makedirs(figures_path, exist_ok=True)


# Prepare features (X) and target (y)
# No one-hot encoding is needed for CatBoost
X = df[feature_cols]
y = df[target_col]
sites = df["site_reference"].unique()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    X[col] = X[col].astype('category')

results = []
all_preds_df_list = []

# Leave-One-Site-Out CV
for test_site in sites:
    print(f"Processing site: {test_site}...")
    train_idx = df["site_reference"] != test_site
    test_idx = df["site_reference"] == test_site

    if test_idx.sum() < 1:
        continue

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    dates_test = df.loc[test_idx, "date"]

    # Initialize and train the CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        subsample=0.7,
        l2_leaf_reg = 1,
        random_state=42,
        cat_features=categorical_features,
        verbose=0, # Suppress verbose output
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

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Site": test_site,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

# Combine results
results_df = pd.DataFrame(results)
all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

# Save to disk with '_cat' suffix
results_csv_path = os.path.join(out_path, f'catboost_results_{target_col}_cat.csv')
predictions_csv_path = os.path.join(out_path, f'catboost_predictions_{target_col}_cat.csv')
results_df.to_csv(results_csv_path, index=False)
all_preds_df.to_csv(predictions_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")
print(f"Predictions saved to: {predictions_csv_path}")


# Pooled metrics
rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])

print("\n--- Site-Specific Results ---")
print(results_df)
print("\n--- Pooled Metrics ---")
print(f"Pooled RMSE: {rmse_all:.4f}")
print(f"Pooled MAE:  {mae_all:.4f}")
print(f"Pooled R²:   {r2_all:.4f}")

# Median metrics across sites
median_rmse = results_df["RMSE"].median()
median_mae = results_df["MAE"].median()
median_r2 = results_df["R2"].median()

print("\n--- Median Metrics Across Sites ---")
print(f"Median RMSE: {median_rmse:.4f}")
print(f"Median MAE:  {median_mae:.4f}")
print(f"Median R²:   {median_r2:.4f}")

# --- Plotting ---
# Loop through each site and save a separate plot
unique_sites = all_preds_df["Site"].unique()
if not unique_sites.any():
    print("\nNo sites to plot.")
else:
    print("\nGenerating and saving individual site plots...")
    for site in unique_sites:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]
        rmse_val = round(site_metrics["RMSE"], 2)
        r2_val = round(site_metrics["R2"], 2)
        mae_val = round(site_metrics["MAE"], 2)

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate() # Auto-formats the x-axis labels for dates

        # Add metrics text to the plot
        textstr = f"RMSE: {rmse_val}\nMAE: {mae_val}\nR²: {r2_val}"
        ax.text(
            0.97, 0.03, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        
        # Define the output path for the plot
        plot_filename = f'catboost_{target_col}_{site}_timeseries_cat.png'
        plot_path = os.path.join(figures_path, plot_filename)
        
        # Save the figure
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Close the plot to free up memory
        plt.close(fig)
        
    print(f"All site plots saved to: {figures_path}")


# 10-fold CH4

# In[ ]:


import os
import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
predictor_vars = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features
]
target_var = 'ch4_flux_total'
categorical_features = ['land_cover', 'month']

# 4. Drop rows only if the target variable is missing
# CatBoost can also handle missing values (NaN) in predictor variables.
df_model = df.dropna(subset=[target_var]).copy()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    df_model[col] = df_model[col].astype('category')

# 5. Define features (X) and target (y)
X = df_model[predictor_vars]
y = df_model[target_var]

# 6. Define parameter grid for CatBoost
# Note: CatBoost uses 'depth' instead of 'max_depth', 'iterations' for 'n_estimators'
param_grid = {
    'depth': [4, 6, 8, 10, 12, 14, 16],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [0.1, 0.5, 1, 3, 5],
    'iterations': [700, 1000, 1500]
}

# 7. Model and 10-fold CV
# Pass the list of categorical features directly to the model
cat_model = CatBoostRegressor(
    random_state=42,
    cat_features=categorical_features,
    verbose=0, # Suppress verbose output during training
    allow_writing_files=False # Suppress creation of catboost_info dir
)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 8. Grid search (optimize RMSE)
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X, y)

# 9. Metrics
best_rmse = -grid_search.best_score_
best_params = grid_search.best_params_
print(f"✅ Best RMSE: {best_rmse:.3f}")
print("✅ Best Parameters:", best_params)

# 10. R² using the best estimator from grid search
r2_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# 11. MAE using the best estimator from grid search
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mae_scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    scoring=mae_scorer,
    cv=cv,
    n_jobs=-1
)
print(f"✅ 10-Fold CV MAE: {-np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")


# In[ ]:


#LOSO CH4


# In[ ]:


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

# 1. Load your dataset
file_path = "/explore/nobackup/people/spotter5/anna_v/v2/v2_model_training_alt_soil_lc_co2.csv"
df = pd.read_csv(file_path)

df = df[df['flux_method'] == 'EC']

# 2. Create tmean_C and date
df['tmean_C'] = df[['tmmn', 'tmmx']].mean(axis=1)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 3. Define predictors and target
# Added 'land_cover' and 'month' to the list of predictors for CatBoost
feature_cols = [
    'EVI', 'NDVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 
    'sur_refl_b07', 'NDWI', 'pdsi', 'srad', 'tmean_C', 'vap', 'vs',
    'bdod_0_100cm', 'cec_0_100cm', 'cfvo_0_100cm', 'clay_0_100cm',
    'nitrogen_0_100cm', 'ocd_0_100cm', 'phh2o_0_100cm', 'sand_0_100cm',
    'silt_0_100cm', 'soc_0_100cm', 'co2_cont', 'ALT',
    'land_cover', 'month' # Categorical features are now included directly
]
target_col = 'ch4_flux_total'
categorical_features = ['land_cover', 'month']

# Drop rows only if the target variable or site_reference is missing.
# CatBoost will handle missing values in the numerical predictor variables.
df = df.dropna(subset=['site_reference', target_col])

# Define output path for CSVs and create it
out_path = os.path.join("/explore/nobackup/people/spotter5/anna_v/v2/loocv", target_col)
os.makedirs(out_path, exist_ok=True)

# Define a separate output path for figures and create it
figures_path = os.path.join(out_path, "figures")
os.makedirs(figures_path, exist_ok=True)


# Prepare features (X) and target (y)
# No one-hot encoding is needed for CatBoost
X = df[feature_cols]
y = df[target_col]
sites = df["site_reference"].unique()

# Convert categorical features to 'category' dtype for CatBoost
for col in categorical_features:
    X[col] = X[col].astype('category')

results = []
all_preds_df_list = []

# Leave-One-Site-Out CV
for test_site in sites:
    print(f"Processing site: {test_site}...")
    train_idx = df["site_reference"] != test_site
    test_idx = df["site_reference"] == test_site

    if test_idx.sum() < 1:
        continue

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    dates_test = df.loc[test_idx, "date"]

    # Initialize and train the CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        subsample=0.7,
        l2_leaf_reg = 1,
        random_state=42,
        cat_features=categorical_features,
        verbose=0, # Suppress verbose output
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

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Site": test_site,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

# Combine results
results_df = pd.DataFrame(results)
all_preds_df = pd.concat(all_preds_df_list, ignore_index=True)

# Save to disk with '_cat' suffix
results_csv_path = os.path.join(out_path, f'catboost_results_{target_col}_cat.csv')
predictions_csv_path = os.path.join(out_path, f'catboost_predictions_{target_col}_cat.csv')
results_df.to_csv(results_csv_path, index=False)
all_preds_df.to_csv(predictions_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")
print(f"Predictions saved to: {predictions_csv_path}")


# Pooled metrics
rmse_all = np.sqrt(mean_squared_error(all_preds_df["Observed"], all_preds_df["Predicted"]))
r2_all = r2_score(all_preds_df["Observed"], all_preds_df["Predicted"])
mae_all = mean_absolute_error(all_preds_df["Observed"], all_preds_df["Predicted"])

print("\n--- Site-Specific Results ---")
print(results_df)
print("\n--- Pooled Metrics ---")
print(f"Pooled RMSE: {rmse_all:.4f}")
print(f"Pooled MAE:  {mae_all:.4f}")
print(f"Pooled R²:   {r2_all:.4f}")

# Median metrics across sites
median_rmse = results_df["RMSE"].median()
median_mae = results_df["MAE"].median()
median_r2 = results_df["R2"].median()

print("\n--- Median Metrics Across Sites ---")
print(f"Median RMSE: {median_rmse:.4f}")
print(f"Median MAE:  {median_mae:.4f}")
print(f"Median R²:   {median_r2:.4f}")

# --- Plotting ---
# Loop through each site and save a separate plot
unique_sites = all_preds_df["Site"].unique()
if not unique_sites.any():
    print("\nNo sites to plot.")
else:
    print("\nGenerating and saving individual site plots...")
    for site in unique_sites:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        site_df = all_preds_df[all_preds_df["Site"] == site].sort_values("Date")
        site_metrics = results_df[results_df["Site"] == site].iloc[0]
        rmse_val = round(site_metrics["RMSE"], 2)
        r2_val = round(site_metrics["R2"], 2)
        mae_val = round(site_metrics["MAE"], 2)

        ax.plot(site_df["Date"], site_df["Observed"], label="Observed", marker="o", linestyle='-', markersize=4)
        ax.plot(site_df["Date"], site_df["Predicted"], label="Predicted", marker="x", linestyle='--', markersize=4)
        ax.set_title(f"Observed vs. Predicted {target_col} for Site: {site}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate() # Auto-formats the x-axis labels for dates

        # Add metrics text to the plot
        textstr = f"RMSE: {rmse_val}\nMAE: {mae_val}\nR²: {r2_val}"
        ax.text(
            0.97, 0.03, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        
        # Define the output path for the plot
        plot_filename = f'catboost_{target_col}_{site}_timeseries_cat.png'
        plot_path = os.path.join(figures_path, plot_filename)
        
        # Save the figure
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Close the plot to free up memory
        plt.close(fig)
        
    print(f"All site plots saved to: {figures_path}")


# In[ ]:


't'


# In[ ]:




