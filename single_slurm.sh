#!/bin/sh
##SBATCH --export=ALL
#SBATCH -N1
##SBATCH --nodelist=gpu018
##SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --time=1-0
##SBATCH --cpus-per-task=15
##SBATCH -G4
#SBATCH -G0
#SBATCH --mem-per-cpu=30G


# python /home/spotter5/andrew/ml_fire_projections/Model_runs/model_xgb_north_crossval_stefano.py 
# python /home/spotter5/andrew/ml_fire_projections/Model_runs/model_xgb_south_crossval_stefano.py 
# python /home/spotter5/anna_v/v2/cat.py 
# python /home/spotter5/anna_v/v2/cat_lag.py 
# python /home/spotter5/anna_v/v2/cat_v2_tuned.py 
# python /home/spotter5/anna_v/v2/cat_v2.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot_no_lc.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot_methane.py 
python /home/spotter5/anna_v/v2/cat_v2_plot_methane_0.py 

# python /home/spotter5/anna_v/v2/cat_v2_plot_lag.py 
# python /home/spotter5/anna_v/v2/cat_16_v2.py 
# python /home/spotter5/anna_v/v2/cat_lag_v2.py 
# python /home/spotter5/anna_v/v2/cat_box_v2.py 
# python /home/spotter5/anna_v/v2/cat_weighted_v2.py 
# python /home/spotter5/anna_v/v2/cat_v2_feature_selection.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot_esa_lc.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot_kyle_lc.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot_abc_lc.py 
# python /home/spotter5/anna_v/v2/cat_v2_plot_bawld_lc.py 














