import numpy as np
import sys

from mlbf_dataLoad_sim import SimDataConfig
from mlbf_dataLoad_exp import ExpDataConfig
from mlbf_dataset import MLBF_Dataset, MLBF_Results, MLBF_Dataset_Tag, Logger

from mlbf_cnn_v1 import mlbf_cnn
from mlbf_fcnet_mmRAPID import mlbf_fcnet
from agile_link import agile_link

#### tf1_test2 - Test using a fixed number of PN beams ####
#       - Otherwise the same as test1

#### Overview of the process with these data structures
#   MLBF_Dataset_Tag            - Dataset tag(s) - Info about the file configurations, sim/test setups, and channel assumptions
#       |
#   SimDataConfig/ExpDataConfig - Loaded data configurator - Info about which data and how much data to use specifically for training and testing
#       |
#   MLBF_Dataset                - Loaded data - the loaded dataset with the given parameters for data selection
#       |
#   mlbf_cnn/mlbf_fcnet/etc     - Beam alignment prediction algorithms - train() and test() with the loaded dataset (note: models are saved into the structure)
#       |
#   MLBF_Results                - Alignment results structure - Save the results into this structure for pkl saving, plotting, etc.
fcnet = "FCnet"
cnn = "CNN"
agilelink = "AgileLink"
csmp = "CS-RSS-MP"

# Fixed combinations of beam indices
use_fixed_beam_num  = True
pn_fixed            = [1, 2, 3]             # Number of PN beams to consider in the combos


#### Prepare the test configuration (Dataset tag with settings and names)
sim_or_exp = "sim" #"exp" #
model = cnn #fcnet #csmp #agilelink #

points_per_label = 20
points_per_label_even = True    # Even out the points per label between sim and exp (144 points per label: 16 per label per channel for sim, 18 per label per channel for exp)
points_per_label_even_sim = 16
points_per_label_even_exp = 18

ncombos = 1
pn_a = True
dftsa_a = True
qpd_a = False

extended_test = False
name_ext = ""

#### Computed constants and default values
if (model == agilelink):
    pn_a = False
    dftsa_a = True
    qpd_a = False
    ncombos = 1
    use_fixed_beam_num = False
if (model == csmp):
    pn_a = True
    dftsa_a = False
    qpd_a = False
    ncombos = 1
    use_fixed_beam_num = False

if extended_test:
    name_ext = "ext"

num_beam_types = np.sum([pn_a, dftsa_a, qpd_a])
if num_beam_types != 2:
    use_fixed_beam_num = False
if use_fixed_beam_num:
    ncombos = len(pn_fixed)


#### Dataset Configuration
if (sim_or_exp == "sim"):
    ## Simulation dataset
    # long_tags_sim = ["SNR 20 dB, alpha=0.7", "SNR 30 dB, alpha=0.7"]        # Tag used for long labels and notes
    # short_tags_sim = ["SNR 20 dB, alpha=0.7", "SNR 30 dB, alpha=0.7"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_sim = ["sim", "sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_sim = [20, 30]                   # List of SNRs used (required for both sims and exps)
    # file_tags_sim = ["",""]             # Tag used for filenames (often unused for sims)
    # dates_sim = ["",""]                # List of dates for the test results (often unused for sims)
    # txidxs_sim = ["",""]               # List of transmit power indices used (experiments only)
    # sim_num_sim = [13, 13]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_sim = [3, 3]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_sim = [False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    # long_tags_sim = ["SNR 15 dB, alpha=4", "SNR 30 dB, alpha=4"]        # Tag used for long labels and notes
    # short_tags_sim = ["mult_sim1_15dB", "mult_sim1_30dB"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_sim = ["sim", "sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_sim = [15, 30]                   # List of SNRs used (required for both sims and exps)
    # file_tags_sim = ["",""]             # Tag used for filenames (often unused for sims)
    # dates_sim = ["",""]                # List of dates for the test results (often unused for sims)
    # txidxs_sim = ["",""]               # List of transmit power indices used (experiments only)
    # sim_num_sim = [1, 1]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_sim = [17.4, 17.4]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_sim = [False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    # long_tags_sim = ["SNR 15 dB, alpha=4", "SNR 15 dB, alpha=0.12"]        # Tag used for long labels and notes
    # short_tags_sim = ["mult_sim1_15dB", "mult_sim9_15dB"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_sim = ["sim", "sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_sim = [15, 15]                   # List of SNRs used (required for both sims and exps)
    # file_tags_sim = ["",""]             # Tag used for filenames (often unused for sims)
    # dates_sim = ["",""]                # List of dates for the test results (often unused for sims)
    # txidxs_sim = ["",""]               # List of transmit power indices used (experiments only)
    # sim_num_sim = [1, 9]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_sim = [17.4, 0.52]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_sim = [False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    # long_tags_sim = ["SNR 15 dB, alpha=4"]        # Tag used for long labels and notes
    # short_tags_sim = ["mult_sim1_15dB"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_sim = ["sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_sim = [15]                   # List of SNRs used (required for both sims and exps)
    # file_tags_sim = [""]             # Tag used for filenames (often unused for sims)
    # dates_sim = [""]                # List of dates for the test results (often unused for sims)
    # txidxs_sim = [""]               # List of transmit power indices used (experiments only)
    # sim_num_sim = [1]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_sim = [17.4]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_sim = [False]  # Boolean flags to indicate which files should only be used for testing (not training)

    # long_tags_sim = ["SNR 30 dB, LOS"]        # Tag used for long labels and notes
    # short_tags_sim = ["mult_sim1_30dB"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_sim = ["sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_sim = [30]                   # List of SNRs used (required for both sims and exps)
    # file_tags_sim = [""]             # Tag used for filenames (often unused for sims)
    # dates_sim = [""]                # List of dates for the test results (often unused for sims)
    # txidxs_sim = [""]               # List of transmit power indices used (experiments only)
    # sim_num_sim = [10]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_sim = [17.4]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_sim = [False]  # Boolean flags to indicate which files should only be used for testing (not training)

    ### FULL DATASET - BOTH SNRS (15 AND 30 dB)
    # long_tags_sim = ["SNR 15 dB, alpha=4.0", "SNR 30 dB, alpha=4.0",\
    #                  "SNR 15 dB, alpha=3.0", "SNR 30 dB, alpha=3.0",\
    #                  "SNR 15 dB, alpha=2.0", "SNR 30 dB, alpha=2.0",\
    #                  "SNR 15 dB, alpha=1.5", "SNR 30 dB, alpha=1.5",\
    #                  "SNR 15 dB, alpha=1.0", "SNR 30 dB, alpha=1.0",\
    #                  "SNR 15 dB, alpha=0.69", "SNR 30 dB, alpha=0.69",\
    #                  "SNR 15 dB, alpha=0.50", "SNR 30 dB, alpha=0.50",\
    #                  "SNR 15 dB, alpha=0.25", "SNR 30 dB, alpha=0.25",\
    #                  "SNR 15 dB, alpha=0.12", "SNR 30 dB, alpha=0.12"]        # Tag used for long labels and notes
    # short_tags_sim = ["mult_sim1_15dB", "mult_sim1_30dB",\
    #                   "mult_sim2_15dB", "mult_sim2_30dB",\
    #                   "mult_sim3_15dB", "mult_sim3_30dB",\
    #                   "mult_sim4_15dB", "mult_sim4_30dB",\
    #                   "mult_sim5_15dB", "mult_sim5_30dB",\
    #                   "mult_sim6_15dB", "mult_sim6_30dB",\
    #                   "mult_sim7_15dB", "mult_sim7_30dB",\
    #                   "mult_sim8_15dB", "mult_sim8_30dB",\
    #                   "mult_sim9_15dB", "mult_sim9_30dB"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_sim = ["sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim",\
    #                   "sim", "sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_sim = [15, 30, 15, 30, 15, 30, 15, 30, 15, 30, 15, 30, 15, 30, 15, 30, 15, 30]         # List of SNRs used (required for both sims and exps)
    # file_tags_sim = ["","", "","", "","", "","", "","", "","", "","", "","", "",""]             # Tag used for filenames (often unused for sims)
    # dates_sim = ["","", "","", "","", "","", "","", "","", "","", "","", "",""]                 # List of dates for the test results (often unused for sims)
    # txidxs_sim = ["","", "","", "","", "","", "","", "","", "","", "","", "",""]                # List of transmit power indices used (experiments only)
    # sim_num_sim = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]                        # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_sim = [17.4, 17.4, 13.0, 13.0, 8.69, 8.69, 6.51, 6.51, 4.34, 4.34, 3.00, 3.00, 2.17, 2.17, 1.09, 1.09, 0.52, 0.52]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_sim = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]  # Boolean flags to indicate which files should only be used for testing (not training)
    # new_filestem = "../data/mult_sim{}"

    ### PARTIAL DATASET (ONLY 15 dB SNR TO BE EQUIVALENT TO THE EXPERIMENTAL DATA)
    long_tags_sim = ["SNR 15 dB, alpha=2.0", \
                     "SNR 15 dB, alpha=1.5", \
                     "SNR 15 dB, alpha=1.0", \
                     "SNR 15 dB, alpha=0.75", \
                     "SNR 15 dB, alpha=0.5", \
                     "SNR 15 dB, alpha=0.35", \
                     "SNR 15 dB, alpha=0.25", \
                     "SNR 15 dB, alpha=0.13", \
                     "SNR 15 dB, alpha=0.06"]        # Tag used for long labels and notes
    short_tags_sim = ["mult_sim1_15dB",\
                      "mult_sim2_15dB",\
                      "mult_sim3_15dB",\
                      "mult_sim4_15dB",\
                      "mult_sim5_15dB",\
                      "mult_sim6_15dB",\
                      "mult_sim7_15dB",\
                      "mult_sim8_15dB",\
                      "mult_sim9_15dB"]           # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    sim_or_exp_sim = ["sim",\
                      "sim",\
                      "sim",\
                      "sim",\
                      "sim",\
                      "sim",\
                      "sim",\
                      "sim",\
                      "sim"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    snrs_sim = [15, 15, 15, 15, 15, 15, 15, 15, 15]         # List of SNRs used (required for both sims and exps)
    file_tags_sim = ["", "", "", "", "", "", "", "", ""]             # Tag used for filenames (often unused for sims)
    dates_sim = ["", "", "", "", "", "", "", "", ""]                 # List of dates for the test results (often unused for sims)
    txidxs_sim = ["", "", "", "", "", "", "", "", ""]                # List of transmit power indices used (experiments only)
    sim_num_sim = [1, 2, 3, 4, 5, 6, 7, 8, 9]                        # Simulation number (-1 for experimental data)
    nlos_rel_pwr_sim = [17.4, 13.0, 8.69, 6.51, 4.34, 3.00, 2.17, 1.09, 0.52]   # NLOS path relative power compared to the stronges/LOS path
    test_only_flags_sim = [False, False, False, False, False, False, False, False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    new_filestem = "../data/mult_sim{}"
    if points_per_label_even:
        points_per_label = points_per_label_even_sim

elif (sim_or_exp == "exp"):
    ## Experiment dataset
    # long_tags_exp = ["SNR 15 dB, alpha=4.61", "SNR 15 dB, alpha=2.30", "SNR 15 dB, alpha=2.30", "SNR 15 dB, alpha=1.15"]        # Tag used for long labels and notes
    # short_tags_exp = ["teslaLOS", "teslaMult_p8n6", "teslaMult_p6n8", "teslaMult_NLOS4"]    # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_exp = ["exp", "exp", "exp", "exp"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_exp = [15, 15, 15, 15]                   # List of SNRs used (required for both sims and exps)
    # file_tags_exp = ["teslaLOS", "teslaMult_p8n6", "teslaMult_p6n8", "teslaMult_NLOS4"]             # Tag used for filenames (often unused for sims)
    # dates_exp = ["21-02-17", "21-02-18", "21-02-18", "21-02-21"]                # List of dates for the test results (often unused for sims)
    # txidxs_exp = ["15", "15", "15", "21"]               # List of transmit power indices used (experiments only)
    # sim_num_exp = [-1, -1, -1, -1]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_exp = [20, 10, 10, 5]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_exp = [False, False, False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    # long_tags_exp = ["SNR 15 dB, alpha=1.43", "SNR 15 dB, alpha=0.83", "SNR 15 dB, alpha=0.23", "SNR 15 dB, alpha=0.12"]        # Tag used for long labels and notes
    # short_tags_exp = ["teslaMult_NLOS9", "teslaMult_NLOS8", "teslaMult_NLOS6", "teslaMult_NLOS7"]    # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    # sim_or_exp_exp = ["exp", "exp", "exp", "exp"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    # snrs_exp = [15, 15, 15, 15]                   # List of SNRs used (required for both sims and exps)
    # file_tags_exp = ["teslaMult_NLOS9", "teslaMult_NLOS8", "teslaMult_NLOS6", "teslaMult_NLOS7"]             # Tag used for filenames (often unused for sims)
    # dates_exp = ["21-04-03", "21-04-02", "21-03-30", "21-03-31"]                # List of dates for the test results (often unused for sims)
    # txidxs_exp = ["27", "27", "27", "27"]               # List of transmit power indices used (experiments only)
    # sim_num_exp = [-1, -1, -1, -1]                  # Simulation number (-1 for experimental data)
    # nlos_rel_pwr_exp = [6.2, 3.6, 1, 0.5]   # NLOS path relative power compared to the stronges/LOS path
    # test_only_flags_exp = [False, False, False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    long_tags_exp = ["SNR 15 dB, alpha=4.61", "SNR 15 dB, alpha=2.30", "SNR 15 dB, alpha=2.30", "SNR 15 dB, alpha=1.15", "SNR 15 dB, alpha=1.43", "SNR 15 dB, alpha=0.83", "SNR 15 dB, alpha=0.23", "SNR 15 dB, alpha=0.12"]        # Tag used for long labels and notes
    short_tags_exp = ["teslaLOS", "teslaMult_p8n6", "teslaMult_p6n8", "teslaMult_NLOS4", "teslaMult_NLOS9", "teslaMult_NLOS8", "teslaMult_NLOS6", "teslaMult_NLOS7"]    # Tag used for indexing results (often SNR for simulation and some short label for experiments)
    sim_or_exp_exp = ["exp", "exp", "exp", "exp", "exp", "exp", "exp", "exp"]         # "sim" or "exp" tags to indicate whether the data is from simulation or experiments respectively
    snrs_exp = [15, 15, 15, 15, 15, 15, 15, 15]                   # List of SNRs used (required for both sims and exps)
    file_tags_exp = ["teslaLOS", "teslaMult_p8n6", "teslaMult_p6n8", "teslaMult_NLOS4", "teslaMult_NLOS9", "teslaMult_NLOS8", "teslaMult_NLOS6", "teslaMult_NLOS7"]             # Tag used for filenames (often unused for sims)
    dates_exp = ["21-02-17", "21-02-18", "21-02-18", "21-02-21", "21-04-03", "21-04-02", "21-03-30", "21-03-31"]                # List of dates for the test results (often unused for sims)
    txidxs_exp = ["15", "15", "15", "21", "27", "27", "27", "27"]               # List of transmit power indices used (experiments only)
    sim_num_exp = [-1, -1, -1, -1, -1, -1, -1, -1]                  # Simulation number (-1 for experimental data)
    nlos_rel_pwr_exp = [20, 10, 10, 5, 6.2, 3.6, 1, 0.5]   # NLOS path relative power compared to the stronges/LOS path
    test_only_flags_exp = [False, False, False, False, False, False, False, False]  # Boolean flags to indicate which files should only be used for testing (not training)

    if points_per_label_even:
        points_per_label = points_per_label_even_exp

test_name = "tf1_test2{}_{}_{}_{}pts_PN{}_QPD{}_DFTSA{}".format(name_ext, model, sim_or_exp, points_per_label, 1*pn_a, 1*qpd_a, 1*dftsa_a)         # Name used for this test configuration and file name (i.e. short description of the comparison/simulation/experiment)
test_title = "Simulated Architecture Comparison"                # Name used for titles of plots (if necessary)


## Generate the actual dataset structure
print("Generating tag(s) for dataset setup...")
if (sim_or_exp == "sim"):
    tag_sim = MLBF_Dataset_Tag(test_name, long_tags_sim, short_tags_sim, sim_or_exp_sim, 
                            snrs_sim, nlos_rel_pwr_sim, 
                            test_only_flags_sim,
                            sim_num_sim, file_tags_sim, dates_sim, txidxs_sim)
    tag = tag_sim
elif (sim_or_exp == "exp"):
    tag_exp = MLBF_Dataset_Tag(test_name, long_tags_exp, short_tags_exp, sim_or_exp_exp, 
                            snrs_exp, nlos_rel_pwr_exp, 
                            test_only_flags_exp,
                            sim_num_exp, file_tags_exp, dates_exp, txidxs_exp)
    tag = tag_exp

#### Other constant configs
# Number of iterations used for training
NN_BATCH_SIZE = 10
NN_NUM_EPOCHS = 30 #NUM_MEAS  # NOTE: THIS HAS NOT BEEN IMPLEMENTED; CHANGE IF NECESSARY

CNN_BATCH_SIZE = 25
CNN_NUM_EPOCHS = 80 # 60 for 60 meas/DFT; 35 for 200 meas/DFT; 70 for 20 meas/DFT

# Print/Plot settings
PLOT_CONFUSION_MATRICES         = False
SAVE_CONFUSION_MATRICES_PLOTS   = False
PLOT_TRAIN_PROCESS              = False     #TODO: ADD THIS FUNCTIONALITY
SAVE_TRAIN_PROCESS              = True      #TODO: ADD THIS FUNCTIONALITY
PLOT_RESULTS                    = False     #TODO: ADD THIS FUNCTIONALITY
PRINT_RESULTS                   = True
PLOT_OVERALL_RESULTS            = False     #TODO: ADD THIS FUNCTIONALITY
SAVE_OVERALL_RESULTS_PLOTS      = True      #TODO: ADD THIS FUNCTIONALITY
SAVE_LOG                        = True
LOG_FILE                        = "./Results/log_{}.txt".format(test_name)

# Setup Logging
if SAVE_LOG:
    sys.stdout = Logger(LOG_FILE)

#### Loop over all desired configs
# Use the full set of combinations desired
if (num_beam_types == 1) and (pn_a == False):
    # Max number of combos and measurements to make this setup work
    all_num_meas        = [      10,       9,       8,       7,       6,       5,       4]
    all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    all_num_beam_combos = [       1, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model]
elif extended_test:
    all_num_meas        = [      36,      34,      32,      30,      28,      26,      24,      22,      20,      18,      16,      14]
    all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    all_num_beam_combos = [ ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model,   model,   model,   model,   model,   model]
    # all_num_meas        = [      28,      26,      24,      22,      20,      18,      16,      14]
    # all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    # all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    # all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    # all_num_beam_combos = [ ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    # all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model,   model]
else:
    # all_num_meas        = [      12,      10,       9,       8,       7,       6,       5,       4]
    # all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    # all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    # all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    # all_num_beam_combos = [ ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    # all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model,   model]
    all_num_meas        = [      12,      11,      10,       9,       8,       7,       6,       5,       4]
    all_use_pn          = [    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a,    pn_a]
    all_use_dftsa       = [ dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a, dftsa_a]
    all_use_qpd         = [   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a,   qpd_a]
    all_num_beam_combos = [ ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos, ncombos]
    all_nn_arch         = [   model,   model,   model,   model,   model,   model,   model,   model,   model]

num_tests = len(all_num_meas)
if use_fixed_beam_num:
    all_use_pn_num = []
    all_use_dftsa_num = []
    all_use_qpd_num = []
    for test_i in np.arange(num_tests):
        all_use_pn_num.append(np.array(pn_fixed))
        if dftsa_a:
            all_use_dftsa_num.append(all_num_meas[test_i] - np.array(pn_fixed))
            all_use_qpd_num.append(np.zeros(len(pn_fixed)))
        if qpd_a:
            all_use_qpd_num.append(all_num_meas[test_i] - np.array(pn_fixed))
            all_use_dftsa_num.append(np.zeros(len(pn_fixed)))

# Initialize structure for the total results of the entire test run
#   Each trained/tested model will provide a results structure, but a copy of this individual 
#   "data point" for the test will be saved to this overall results structure
print("Initializing results structure...")
total_results = MLBF_Results(test_name=test_name, meas_combo_labels=[], max_num_meas=12, results_dir="./Results")
#TODO: FIX THE COMBO LABELS (NEED TO PULL THE BEST SET LABEL FROM EACH NUM MEAS)

for test_i in np.arange(num_tests):
    #### Load the data from the test configuration (MLBF_Dataset produced by the [Sim/Exp]DataConfig functions)
    print("##### TEST {}/{} #####".format(test_i+1, num_tests))
    # Test configs
    num_meas = all_num_meas[test_i]
    use_pn = all_use_pn[test_i]
    use_dftsa = all_use_dftsa[test_i]
    use_qpd = all_use_qpd[test_i]
    num_beam_combos = all_num_beam_combos[test_i]
    nn_arch = all_nn_arch[test_i] # "FCnet" or "CNN"

    # Data loading
    print("## Loading data...")
    if (nn_arch == cnn) or (nn_arch == fcnet):
        if (sim_or_exp == "sim"):
            data_config = SimDataConfig(dataset_tag=tag_sim, 
                                        num_meas=num_meas, num_beam_combos=num_beam_combos, nn_arch=nn_arch,
                                        use_pn=use_pn, use_dftsa=use_dftsa, use_qpd=use_qpd, 
                                        num_meas_per_dft=points_per_label,
                                        data_dir_filestem=new_filestem,
                                        use_linear_scale=True, use_zero_mean=False, use_normalization=True)
        elif (sim_or_exp == "exp"):
            data_config = ExpDataConfig(dataset_tag=tag_exp, 
                                        num_meas=num_meas, num_beam_combos=num_beam_combos, nn_arch=nn_arch,
                                        use_pn=use_pn, use_dftsa=use_dftsa, use_qpd=use_qpd, 
                                        num_meas_per_dft=points_per_label,
                                        use_linear_scale=True, use_zero_mean=False, use_normalization=True)
    elif (nn_arch == agilelink) or (nn_arch == csmp):
        if (sim_or_exp == "sim"):
            data_config = SimDataConfig(dataset_tag=tag_sim, 
                                        num_meas=num_meas, num_beam_combos=num_beam_combos, nn_arch=nn_arch,
                                        use_pn=use_pn, use_dftsa=use_dftsa, use_qpd=use_qpd, 
                                        num_meas_per_dft=points_per_label,
                                        data_dir_filestem=new_filestem,
                                        use_linear_scale=True, use_zero_mean=False, use_normalization=False)
        elif (sim_or_exp == "exp"):
            data_config = ExpDataConfig(dataset_tag=tag_exp, 
                                        num_meas=num_meas, num_beam_combos=num_beam_combos, nn_arch=nn_arch,
                                        use_pn=use_pn, use_dftsa=use_dftsa, use_qpd=use_qpd, 
                                        num_meas_per_dft=points_per_label,
                                        use_linear_scale=True, use_zero_mean=False, use_normalization=False)
        
    print("## Building dataset...")
    if use_fixed_beam_num:
        print("\tUsing the following numbers of beams:")
        print("\t\tPN:    {}".format(all_use_pn_num[test_i]))
        print("\t\tDFTSA: {}".format(all_use_dftsa_num[test_i]))
        print("\t\tQPD:   {}".format(all_use_qpd_num[test_i]))
        data_config.specify_combos(all_use_pn_num[test_i], all_use_dftsa_num[test_i], all_use_qpd_num[test_i])
    data_selected = data_config.create_datasets(plot_meas_nums=False, print_summaries=True)


    #### Setup the algorithms (Initialize the alignment algorithm(s) of choice)
    print("## Setting up model/algorithm...")
    if (nn_arch == cnn): 
        model = mlbf_cnn(data_config.NUM_CLASSES, num_meas, num_beam_combos, tag,
                         BATCH_SIZE=CNN_BATCH_SIZE, NUM_EPOCHS=CNN_NUM_EPOCHS)
    elif (nn_arch == fcnet):
        model = mlbf_fcnet(data_config.NUM_CLASSES, num_meas, num_beam_combos, tag,
                           BATCH_SIZE=NN_BATCH_SIZE, NUM_EPOCHS=NN_NUM_EPOCHS)
    elif (nn_arch == agilelink):
        if (sim_or_exp == 'sim'):
            codebook_file = '../data/Codebooks/awv_dft64_pn36_qpd10_dftsa10_sim.csv'
        elif (sim_or_exp == 'exp'):
            codebook_file = '../data/Codebooks/awv_dft64_pn36_qpd10_dftsa10.csv'
        model = agile_link(data_config.NUM_CLASSES, num_meas, num_beam_combos, tag, 
                           data_config.DFTSA_USE_BEAMS, data_config.dft_use, beam_type='dftsa', 
                           codebook_file=codebook_file, ant_d=0.57)
    elif (nn_arch == csmp):
        if (sim_or_exp == 'sim'):
            codebook_file = '../data/Codebooks/awv_dft64_pn36_qpd10_dftsa10_sim.csv'
        elif (sim_or_exp == 'exp'):
            codebook_file = '../data/Codebooks/awv_dft64_pn36_qpd10_dftsa10.csv'
        model = agile_link(data_config.NUM_CLASSES, num_meas, num_beam_combos, tag, 
                           data_config.PN_USE_BEAMS, data_config.dft_use, beam_type='cs', 
                           codebook_file=codebook_file, ant_d=0.57)


    #### Train the algorithms (use the train() function for the algorithm(s))
    print("## Training model...")
    #import pdb; pdb.set_trace()
    model.train(data_selected, PLOT_PROGRESS=PLOT_TRAIN_PROCESS)


    #### Test the algorithms (use the test() function for the algorithms(s))
    print("## Testing model...")
    model.test(data_selected, PRINT_RESULTS=PRINT_RESULTS, PLOT_RESULTS=PLOT_RESULTS,  
               PLOT_CONFUSION_MATRICES=PLOT_CONFUSION_MATRICES, SAVE_CONFUSION_MATRICES=SAVE_CONFUSION_MATRICES_PLOTS)


    #### Save the results from the algorithms into the MLBF_Results structure, with plots and data files
    #       - Note: models generally should not be saved, since they will likely take up a large amount of data and can readily be retrained
    #model.all_results.set_tag(tag=tag, set_test_name=True)
    print("## Storing results...")
    total_results.copy_results(model.all_results)
    

#import pdb; pdb.set_trace()
total_results.pkl_save_results()
if PLOT_OVERALL_RESULTS:
    total_results.plot_results(SAVE_PLOTS=SAVE_OVERALL_RESULTS_PLOTS)

