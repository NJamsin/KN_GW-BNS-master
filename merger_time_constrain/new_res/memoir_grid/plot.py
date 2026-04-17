import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from peket.kn_side.utils import plot_param_evolution

######################################
# DO NOT FORGET TO UPDATE THE DIFF VAR
######################################
dirs = [#f"/home/stu_jamsin/memoir_grid/ka17_opt", f"/home/stu_jamsin/memoir_grid/bu19_opt", f"/home/stu_jamsin/memoir_grid/bu26_opt",
        #f"/home/stu_jamsin/memoir_grid/ka17_real", f"/home/stu_jamsin/memoir_grid/bu19_real", f"/home/stu_jamsin/memoir_grid/bu26_real"] # change as needed
        f"/home/stu_jamsin/memoir_grid/ka17_copy", f"/home/stu_jamsin/memoir_grid/bu19_copy", f"/home/stu_jamsin/memoir_grid/bu26_copy"] # change as needed
models = [#'Ka2017', 'Bu2019lm', 'Bu2026_MLP',
           'Ka2017', 'Bu2019lm', 'Bu2026_MLP'] # change as needed (keep the same order as the dirs)
# control shape of plot
#col_num = 5
#row_num = 5 
col_num = 3
row_num = 1
UL = False # if UL are present in the data

true_merger = '2020-01-07T00:00:00.000' # change as needed (keep the same for all the analyses to see how the timeshift evolves)
#minus_num = 9
minus_num = 5 # max number of removed points + 1 (to include the full data analysis as well)
#ts_max = -2.5
ts_max = -8.5

for DIR, MODEL in zip(dirs, models):
    plot_param_evolution(DIR=DIR, MODEL=MODEL, col_num=col_num, row_num=row_num, minus_num=minus_num, true_merger=true_merger, UL=UL, ts_max=ts_max)
