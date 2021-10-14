import numpy as np
import matplotlib.pyplot as plt
import sys

from mlbf_dataLoad_sim import SimDataConfig
from mlbf_dataLoad_exp import ExpDataConfig
from mlbf_dataset import MLBF_Dataset, MLBF_Results, MLBF_Dataset_Tag, Logger

from mlbf_cnn_v1 import mlbf_cnn
from mlbf_fcnet_mmRAPID import mlbf_fcnet
from agile_link import agile_link

#### Script to plot the results from tf1_test1.py
#   Pseudocode:
#       - Set the list of result names/parameters
#       - Initialize a unified results structure
#       - Loop through each test configuration
#           - Load the data from each test configuration
#           - Copy the data into the total results structure
#       - Plot the results using the plot function in the total results structure

#### General Constants
# Model names
fcnet = "FCnet"
cnn = "CNN"
agilelink = "AgileLink"
csmp = "CS-RSS-MP"

# Name used for this test configuration and file name (i.e. short description of the comparison/simulation/experiment)
test_name_total = "tf1_test1_total"
test_title = "Simulated Architecture Comparison"                # Name used for titles of plots (if necessary)

test_name_stem = "tf1_test{}{}_{}_{}_{}pts_PN{}_QPD{}_DFTSA{}" #.format(test_num, test_ext, model, sim_or_exp, points_per_label, 1*pn_a, 1*qpd_a, 1*dftsa_a) 
# Test #1 used random beam combos (with a specific number of total meausrements)
# Test #2 used fixed numbers of each beam type (i.e. 1 PN beam + remaining as DFTSA beams)

# Print/Plot settings
SAVE_OVERALL_RESULTS_PLOTS      = False  
SAVE_PLOT_TYPE                  = "pdf" #"png" #"eps" #
SET_DEFAULT_PLOT_SIZE           = True
USE_TIGHT_LAYOUT                = True
## Plot sizes for the wide plot format
PLOT_WIDTH_DEFAULT              = 6
PLOT_HEIGHT_DEFAULT             = 3.5 #2.5 #2.75 #
## Plot sizes for the tall plot format
# PLOT_WIDTH_DEFAULT              = 2.75
# PLOT_HEIGHT_DEFAULT             = 4 #2.75 #3 #

run_sets                        = [2] #[1, 2, 3, 4] #       # Set of plots to generate
# Plot set 1: 1.1. - Gain loss comparison of models
# Plot set 2: 1.2. - Gain loss comparison of beam designs/features
# Plot set 3: 3.3. - Required number of measurements for set of notable designs (model + beam combo)
# Plot set 4: 3.2. - Required number of measurements for the notable designs but in bargraph form

plot_width_set = -1
plot_height_set = -1
if SET_DEFAULT_PLOT_SIZE:
    plot_width_set = PLOT_WIDTH_DEFAULT
    plot_height_set = PLOT_HEIGHT_DEFAULT

#### Test List - Each test gets an entry in each list of configurations
#   - Defines the parameters used in the filenames/results structures

## Old, non-extended
# sim_or_exp_all  = [     "sim",     "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp"]
# model_all       = [ agilelink, agilelink,  csmp,  csmp,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet]
# ptsPerLabel_all = [        20,        20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20]
# ncombos_all     = [         1,         1,     1,     1,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3]
# pn_a_all        = [     False,     False,  True,  True, False, False,  True,  True,  True,  True,  True,  True, False, False,  True,  True,  True,  True,  True,  True]
# qpd_a_all       = [     False,     False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False,  True,  True]
# dftsa_a_all     = [      True,      True, False, False,  True,  True, False, False,  True,  True, False, False,  True,  True, False, False,  True,  True, False, False]
# test_num_all    = [         1,         1,     1,     1,     1,     1,     1,     1,     2,     2,     1,     1,     1,     1,     1,     1,     2,     2,     1,     1]

## Old, extended included
# sim_or_exp_all  = [     "sim",     "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp"]
# model_all       = [ agilelink, agilelink,  csmp,  csmp,  csmp,  csmp,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet]
# ptsPerLabel_all = [        20,        20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20,    20]
# ncombos_all     = [         1,         1,     1,     1,     1,     1,     3,     3,     3,     3,     3,     3,     3,     3,     1,     1,     3,     3,     3,     3,     3,     3,     3,     3,     1,     1]
# pn_a_all        = [     False,     False,  True,  True,  True,  True, False, False,  True,  True,  True,  True,  True,  True,  True,  True, False, False,  True,  True,  True,  True,  True,  True,  True,  True]
# qpd_a_all       = [     False,     False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False,  True,  True, False, False]
# dftsa_a_all     = [      True,      True, False, False, False, False,  True,  True, False, False,  True,  True, False, False, False, False,  True,  True, False, False,  True,  True, False, False, False, False]
# test_num_all    = [         1,         1,     1,     1,     2,     2,     1,     1,     1,     1,     2,     2,     1,     1,     2,     2,     1,     1,     1,     1,     2,     2,     1,     1,     2,     2]
# extended_all    = [     False,     False, False, False,  True,  True, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False,  True,  True] 

## Updated (fixed sims),  extended included
sim_or_exp_all  = [     "sim",     "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp", "sim", "exp"]
model_all       = [ agilelink, agilelink,  csmp,  csmp,  csmp,  csmp,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn,   cnn, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet, fcnet]
ptsPerLabel_all = [        16,        18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18,    16,    18]
ncombos_all     = [         1,         1,     1,     1,     1,     1,     1,     1,     1,     1,     3,     3,     3,     3,     1,     1,     1,     1,     1,     1,     3,     3,     3,     3,     1,     1]
pn_a_all        = [     False,     False,  True,  True,  True,  True, False, False,  True,  True,  True,  True,  True,  True,  True,  True, False, False,  True,  True,  True,  True,  True,  True,  True,  True]
qpd_a_all       = [     False,     False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False,  True,  True, False, False]
dftsa_a_all     = [      True,      True, False, False, False, False,  True,  True, False, False,  True,  True, False, False, False, False,  True,  True, False, False,  True,  True, False, False, False, False]
test_num_all    = [         2,         2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2]
extended_all    = [     False,     False, False, False,  True,  True, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False,  True,  True] 

num_results = len(sim_or_exp_all)


#### Initialize structure for the total results of the entire test run
#   Each trained/tested model will provide a results structure, but a copy of this individual 
#   "data point" for the test will be saved to this overall results structure
print("Initializing total results structure...")
total_results = MLBF_Results(test_name=test_name_total, meas_combo_labels=[], max_num_meas=12, results_dir="./Results")

#### Loop through each test configuration
for result_i in np.arange(num_results):
    print("##### RESULT {}/{} #####".format(result_i+1, num_results))

    ## Prepare the test configuration (Dataset tag with settings and names)
    sim_or_exp = sim_or_exp_all[result_i]
    model = model_all[result_i]
    points_per_label = ptsPerLabel_all[result_i]
    ncombos = ncombos_all[result_i]
    pn_a = pn_a_all[result_i]
    dftsa_a = dftsa_a_all[result_i]
    qpd_a = qpd_a_all[result_i]
    test_num_i = test_num_all[result_i]
    extended_i = extended_all[result_i]

    # Restriction enforced in the test code; should not do anything since this info should be reflected in the test list
    if (model == agilelink):
        pn_a = False
        dftsa_a = True
        qpd_a = False
        ncombos = 1
    elif (model == csmp):
        pn_a = True
        dftsa_a = False
        qpd_a = False
        ncombos = 1

    num_beam_types = np.sum([pn_a, dftsa_a, qpd_a])
    test_ext_i = ""
    if extended_i:
        test_ext_i = "ext"
    test_name = test_name_stem.format(test_num_i, test_ext_i, model, sim_or_exp, points_per_label, 1*pn_a, 1*qpd_a, 1*dftsa_a)         # Name used for this test configuration and file name (i.e. short description of the comparison/simulation/experiment)
    temp_results = MLBF_Results(test_name=test_name, meas_combo_labels=[], max_num_meas=12, results_dir="./Results")


    # ## Loop over all desired configs
    # # Use the full set of combinations desired
    # if (num_beam_types == 1) and (pn_a == False):
    #     # Max number of combos and measurements to make this setup work
    #     all_num_meas        = [      10,       9,       8,       7,       6,       5,       4]
    #     all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    #     all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    #     all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    #     all_num_beam_combos = [       1, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    #     all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model]
    # else:
    #     all_num_meas        = [      12,      10,       9,       8,       7,       6,       5,       4]
    #     all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    #     all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    #     all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    #     all_num_beam_combos = [ ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    #     all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model,   model]

    ## Load the data from the test configuration (MLBF_Dataset produced by the [Sim/Exp]DataConfig functions)
    temp_results.pkl_load_results()
    # Copy the results into the total results structure
    total_results.copy_results(temp_results)
        

#import pdb; pdb.set_trace()

#### Plot the combined results
# total_results.pkl_save_results()

#  Plot metrics:
#   1. Gain loss (DFT gain for the best beam - gain for the selected beam)
#   2. Test accuracy
#   3. Required number of measurements vs. alpha (used with plot types 1)
#  Plot types: (plot metric vs number of measurements unless otherwise listed)
#   1. Comparison of models (fixed channel characteristics and measurement combo)
#   2. Comparison of measurement combos (fixed channel characteristics and model)
#   3. Comparison of channel performance (fixed measurement combo and model)
#   4. 


## Plots comparing model architectures and sim/exp - fixed channel, beam combo
##      - 90th percentile gain loss vs number of measurements

#### Plot parameters ####
fixed_rel_pwr_weak_sim = 17.4
fixed_rel_pwr_weak_exp = 20

# fixed_rel_pwr_med_sim = 6.51
# fixed_rel_pwr_med_exp = 6.2
fixed_rel_pwr_med_sim = 4.34
fixed_rel_pwr_med_exp = 3.6

gain_loss_perc = 90
req_gain_loss = 3.0

## Selected NLOS levels for plot type 4 (3.2.) - ORDER MATTERS (DETERMINES ORDER ON PLOT)
#   All sim NLOS levels: [17.4, 13.0, 8.69, 6.51, 4.34, 3.00, 2.17, 1.09, 0.52]
bar_nlos_rel_pwr_sim = [1.09, 8.69, 17.4]
# bar_nlos_rel_pwr_sim = [1.09, 4.34, 17.4]
#   All exp NLOS levels: [20, 10, 10, 5, 3.6, 1, 0.5]
bar_nlos_rel_pwr_exp = [1, 10, 20]
# bar_nlos_rel_pwr_exp = [1, 5, 20]

# Plots comparing the models - fixed channel and features
if 1 in run_sets:
    total_results.plot_results(plot_metric=1, plot_type=1, 
                            test_title="tf1_test2_compModel_90perc_multWeak_1pn-sa", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_rel_pwr_sim=fixed_rel_pwr_weak_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_weak_exp, fixed_channel_snr=15, fixed_meas_combo="1 PN Beams, [0-9]* SA Beams",
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)

    total_results.plot_results(plot_metric=1, plot_type=1, 
                            test_title="tf1_test2_compModel_90perc_multMed_1pn-sa", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_rel_pwr_sim=fixed_rel_pwr_med_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_med_exp, fixed_channel_snr=15, fixed_meas_combo="1 PN Beams, [0-9]* SA Beams",
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)

    ## TODO FIX ISSUE WITH THIS PLOT
    # total_results.plot_results(plot_metric=1, plot_type=1, 
    #                         test_title="tf1_test2_compModel_90perc_multMed_exp_1pn-qpd", sim_exp="exp",
    #                         gain_loss_perc=90,
    #                         fixed_model="CNN", fixed_channel_rel_pwr_sim=6.51, fixed_channel_rel_pwr_exp=6.2, fixed_channel_snr=15, fixed_meas_combo="1 PN Beams, [0-9] QPD Beams",
    #                         SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE)

    total_results.plot_results(plot_metric=1, plot_type=1, 
                            test_title="tf1_test2_compModel_90perc_multMed_pnOnly", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_rel_pwr_sim=fixed_rel_pwr_med_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_med_exp, fixed_channel_snr=15, fixed_meas_combo="[0-9]* PN Beams\Z",
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)

    total_results.plot_results(plot_metric=1, plot_type=1, 
                            test_title="tf1_test2_compModel_90perc_multMed_saOnly", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_rel_pwr_sim=fixed_rel_pwr_med_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_med_exp, fixed_channel_snr=15, fixed_meas_combo="\A[0-9]* SA Beams",
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)



## Plots comparing features - fixed channel and architecture (CNN) 
if 2 in run_sets:
    print("\tPlotting #1...")
    total_results.plot_results(plot_metric=1, plot_type=2, 
                            test_title="tf1_test2_compBeams_90perc_multWeak_exp_1pn-sa", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_model="CNN", fixed_channel_rel_pwr_sim=fixed_rel_pwr_weak_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_weak_exp, fixed_channel_snr=15, 
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)

    print("\tPlotting #2...")
    total_results.plot_results(plot_metric=1, plot_type=2, 
                            test_title="tf1_test2_compBeams_90perc_multWeak_sim_1pn-sa", sim_exp="sim",
                            gain_loss_perc=gain_loss_perc,
                            fixed_model="CNN", fixed_channel_rel_pwr_sim=fixed_rel_pwr_weak_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_weak_exp, fixed_channel_snr=15, 
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)

    print("\tPlotting #3...")
    total_results.plot_results(plot_metric=1, plot_type=2, 
                            test_title="tf1_test2_compBeams_90perc_multMed_exp_1pn-sa", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_model="CNN", fixed_channel_rel_pwr_sim=fixed_rel_pwr_med_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_med_exp, fixed_channel_snr=15, 
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)

    print("\tPlotting #4...")
    total_results.plot_results(plot_metric=1, plot_type=2, 
                            test_title="tf1_test2_compBeams_90perc_multMed_sim_1pn-sa", sim_exp="sim",
                            gain_loss_perc=gain_loss_perc,
                            fixed_model="CNN", fixed_channel_rel_pwr_sim=fixed_rel_pwr_med_sim, fixed_channel_rel_pwr_exp=fixed_rel_pwr_med_exp, fixed_channel_snr=15, 
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)



## Plots comparing the required number of measurements
if 3 in run_sets:
    print("\tPlotting #1...")
    total_results.plot_results(plot_metric=3, plot_type=3, 
                            test_title="tf1_test2_reqMeas_90perc_exp", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_snr=15, max_gain_loss=req_gain_loss,
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)
    print("\tPlotting #2...")
    total_results.plot_results(plot_metric=3, plot_type=3, 
                            test_title="tf1_test2_reqMeas_90perc_sim", sim_exp="sim",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_snr=15, max_gain_loss=req_gain_loss,
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)



## Plots comparing the required number of measurements - BAR GRAPH VERSION
if 4 in run_sets:
    print("\tPlotting #1...")
    total_results.plot_results(plot_metric=3, plot_type=2, 
                            test_title="tf1_test2_reqMeasBar_90perc_exp", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_snr=15, max_gain_loss=req_gain_loss, bar_nlos_rel_pwrs=bar_nlos_rel_pwr_exp,
                            SAVE_PLOTS=False, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)
    print("\tPlotting #2...")
    total_results.plot_results(plot_metric=3, plot_type=2, 
                            test_title="tf1_test2_reqMeasBar_90perc_sim", sim_exp="sim",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_snr=15, max_gain_loss=req_gain_loss, bar_nlos_rel_pwrs=bar_nlos_rel_pwr_sim,
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)
    print("\tPlotting #1...")
    total_results.plot_results(plot_metric=3, plot_type=2, 
                            test_title="tf1_test2_reqMeasBar_90perc_exp", sim_exp="exp",
                            gain_loss_perc=gain_loss_perc,
                            fixed_channel_snr=15, max_gain_loss=req_gain_loss, bar_nlos_rel_pwrs=bar_nlos_rel_pwr_exp,
                            SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS, plot_width=plot_width_set, plot_height=plot_height_set, save_type=SAVE_PLOT_TYPE,
                            plot_tight_layout=USE_TIGHT_LAYOUT)


plt.show()