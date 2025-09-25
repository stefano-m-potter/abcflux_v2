import subprocess
import os
import numpy as np



# years = [2001, 2005, 2008, 2017, 2018, 2019, 2020, 2021, 2022]

years = range(2025)
# years = [2013]


# #for var in all_variables:
for i in years:   
        #        for rcp in rcps:        
    j_name = "-J {}".format(i)
    #        print (j_name)
    subprocess.call(["sbatch", j_name, "single_slurm.sh", str(i)])


# for year in years:   
#     j_name = "-J {}".format(year)
#     out_log = "-o out_{}.log".format(year)  # Create a unique log file for each year
#     subprocess.call(["sbatch", j_name, out_log, "single_slurm.sh", str(year)])
