import numpy as np
import os
import pickle
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import sys

## Widely used constants
GAIN_LOSS_PERCENTILES = np.arange(0, 110, 10)
# Model names
fcnet = "FCnet"
cnn = "CNN"
agilelink = "AgileLink"
csmp = "CS-RSS-MP"

## Dataset class
class MLBF_Dataset:
    #### Dataset class (holds loaded data)
    #       Note that these objects are generally NOT saved - just used to readily organized data
    #       Data should be stored in their original CSV files and loaded using the following classes:
    #           - Simulation results:   SimDataConfig (mlbf_dataLoad_sim.py)
    #           - Experimental results: ExpDataConfig (mlbf_dataLoad_exp.py)
    
    def __init__(self, sim_or_exp, config):
        ## Standard dictionaries (FCnet and traditional algorithms)
        self.train_data_dict = {}
        self.test_data_dict = {}
        self.val_data_dict = {}
        self.train_labels_dict = {}
        self.test_labels_dict = {}
        self.val_labels_dict = {}
        self.train_classes_dict = {}
        self.test_classes_dict = {}
        self.val_classes_dict = {}
        self.dft_rssi_dict = {}

        self.train_data_dict_r = {}
        self.train_labels_dict_r = {}
        self.train_classes_dict_r = {}
        self.val_data_dict_r = {}
        self.val_labels_dict_r = {}
        self.val_classes_dict_r = {}
        self.test_data_dict_r = {}
        self.test_labels_dict_r = {}
        self.test_classes_dict_r = {}

        if (sim_or_exp == "sim") or (sim_or_exp == "exp"):
            self.data_type = sim_or_exp
        else:
            print("ERROR: Invalid selection for data type ({}) - must be \"sim\" or \"exp\"".format(sim_or_exp))

        # Should be a SimDataConfig or ExpDataConfig object
        self.Config = config
    

    def store_train_data(self, train_data_dict, train_labels_dict, train_classes_dict):
        self.train_data_dict = train_data_dict
        self.train_labels_dict = train_labels_dict
        self.train_classes_dict = train_classes_dict


    def store_test_data(self, test_data_dict, test_labels_dict, test_classes_dict, test_dftrssi_dict):
        self.test_data_dict = test_data_dict
        self.test_labels_dict = test_labels_dict
        self.test_classes_dict = test_classes_dict
        self.dft_rssi_dict = test_dftrssi_dict


    def store_val_data(self, val_data_dict, val_labels_dict, val_classes_dict):
        self.val_data_dict = val_data_dict
        self.val_labels_dict = val_labels_dict
        self.val_classes_dict = val_classes_dict


    def convert_for_cnn(self):
        ## Save an additional copy of the data in the proper format for CNN evaluation
        #   Adds an extra dimension to the arrays (required for CNN functions)
        #   Saved separately so regular FC-Nets can still be evaluated for comparison
        #
        #   TODO: Add support for validation datasets (typically not used for our work so far)

        ## Compute the constants
        NUM_PN_COMBO = self.Config.NUM_PN_COMBO
        DATA_SNR = self.Config.DATA_SNR

        ## Storage dictionaries/running variables for regression
        train_data_dict_r = {}
        test_data_dict_r = {}
        # val_data_dict_r = {}
        train_classes_dict_r = {}
        test_classes_dict_r = {}
        # val_classes_dict_r = {}
        train_labels_dict_r = {}
        test_labels_dict_r = {}
        # val_labels_dict_r = {}

        train_data_all_r = {}
        test_data_all_r = {}
        # val_data_all_r = {}
        train_classes_all_r = {}
        test_classes_all_r = {}
        # val_classes_all_r = {}
        train_labels_all_r = {}
        test_labels_all_r = {}
        # val_labels_dict_r = {}


        ## Manage the data formats
        # For the regression method with CNNs, need dimensions to be N x D x 1 (instead of N x D or D x N)
        print("Modifying training/validation/testing datasets:")
        for snr_ind in np.arange(len(DATA_SNR)):
            if snr_ind != len(DATA_SNR):
                SNR_i = DATA_SNR[snr_ind]
                print("{} dB SNR --".format(SNR_i))
            else:
                SNR_i = 'ALL'
                print("ALL SNR values --")
                
            train_data_dict_r_temp = {}
            test_data_dict_r_temp = {}
        #     val_data_dict_r_temp = {}
            train_classes_dict_r_temp = {}
            test_classes_dict_r_temp = {}
        #     val_classes_dict_r_temp = {}
            train_labels_dict_r_temp = {}
            test_labels_dict_r_temp = {}
        #     val_labels_dict_r_temp = {}

            for pn_i in np.arange(NUM_PN_COMBO):
                # Save to the temporary for the given PN set
                train_data_dict_r_temp[pn_i] = np.array([self.train_data_dict[snr_ind][pn_i].T]).T
                train_labels_dict_r_temp[pn_i] = np.array([self.train_labels_dict[snr_ind][pn_i].T]).T
                train_classes_dict_r_temp[pn_i] = np.array([self.train_classes_dict[snr_ind][pn_i].T]).T

        #         val_data_dict_r_temp[pn_i] = np.array([val_data_dict[snr_ind][pn_i].T]).T
        #         val_labels_dict_r_temp[pn_i] = np.array([val_labels_dict[snr_ind][pn_i].T]).T
        #         val_classes_dict_r_temp[pn_i] = np.array([val_classes_dict[snr_ind][pn_i].T]).T

                test_data_dict_r_temp[pn_i] = np.array([self.test_data_dict[snr_ind][pn_i].T]).T
                test_labels_dict_r_temp[pn_i] = np.array([self.test_labels_dict[snr_ind][pn_i].T]).T
                test_classes_dict_r_temp[pn_i] = np.array([self.test_classes_dict[snr_ind][pn_i].T]).T
                
            # Add to the total dictionary
            train_data_dict_r[snr_ind] = train_data_dict_r_temp
            train_labels_dict_r[snr_ind] = train_labels_dict_r_temp
            train_classes_dict_r[snr_ind] = train_classes_dict_r_temp

        #     val_data_dict_r[snr_ind] = val_data_dict_r_temp
        #     val_labels_dict_r[snr_ind] = val_labels_dict_r_temp
        #     val_classes_dict_r[snr_ind] = val_classes_dict_r_temp

            test_data_dict_r[snr_ind] = test_data_dict_r_temp
            test_labels_dict_r[snr_ind] = test_labels_dict_r_temp
            test_classes_dict_r[snr_ind] = test_classes_dict_r_temp
            

        # Resize the combined dictionaries (all SNRs) too
        for pn_i in np.arange(NUM_PN_COMBO):
            train_data_all_r[pn_i] = np.array([self.train_data_dict['ALL'][pn_i].T]).T
            train_classes_all_r[pn_i] = np.array([self.train_classes_dict['ALL'][pn_i].T]).T
            train_labels_all_r[pn_i] = np.array([self.train_labels_dict['ALL'][pn_i].T]).T

            # val_data_all_r[pn_i] = np.array([self.val_data_dict['ALL'][pn_i].T]).T
            # val_classes_all_r[pn_i] = np.array([self.val_classes_dict['ALL'][pn_i].T]).T
            # val_labels_dict_r[pn_i] = np.array([self.val_labels_dict['ALL'][pn_i].T]).T

            test_data_all_r[pn_i] = np.array([self.test_data_dict['ALL'][pn_i].T]).T
            test_classes_all_r[pn_i] = np.array([self.test_classes_dict['ALL'][pn_i].T]).T
            test_labels_all_r[pn_i] = np.array([self.test_labels_dict['ALL'][pn_i].T]).T

        # Save the total data to a new dictionary key (useful for test loops)
        train_data_dict_r['ALL'] = train_data_all_r
        train_labels_dict_r['ALL'] = train_labels_all_r
        train_classes_dict_r['ALL'] = train_classes_all_r
        
        # val_data_dict_r['ALL'] = val_data_all_r
        # val_labels_dict_r['ALL'] = val_labels_all_r
        # val_classes_dict_r['ALL'] = val_classes_all_r

        test_data_dict_r['ALL'] = test_data_all_r
        test_labels_dict_r['ALL'] = test_labels_all_r
        test_classes_dict_r['ALL'] = test_classes_all_r
            
        #         train_data_dict_r[snr_ind][pn_i] = np.array([train_data_dict[snr_ind][pn_i].T]).T
        #         train_labels_dict_r[snr_ind][pn_i] = np.array([train_labels_dict[snr_ind][pn_i].T]).T
        #         train_angles_dict_r[snr_ind][pn_i] = np.array([train_angles_dict[snr_ind][pn_i].T]).T

        #         val_data_dict_r[snr_ind][pn_i] = np.array([val_data_dict[snr_ind][pn_i].T]).T
        #         val_labels_dict_r[snr_ind][pn_i] = np.array([val_labels_dict[snr_ind][pn_i].T]).T
        #         val_angles_dict_r[snr_ind][pn_i] = np.array([val_angles_dict[snr_ind][pn_i].T]).T

        #         test_data_dict_r[snr_ind][pn_i] = np.array([test_data_dict[snr_ind][pn_i].T]).T
        #         test_labels_dict_r[snr_ind][pn_i] = np.array([test_labels_dict[snr_ind][pn_i].T]).T
        #         test_angles_dict_r[snr_ind][pn_i] = np.array([test_angles_dict[snr_ind][pn_i].T]).T

        # Overwrite the old data dictionaries for the new ones (Used specifically for CNNs)
        self.train_data_dict_r = train_data_dict_r
        self.train_labels_dict_r = train_labels_dict_r
        self.train_classes_dict_r = train_classes_dict_r

        # self.val_data_dict = val_data_dict_r
        # self.val_labels_dict = val_labels_dict_r
        # self.val_classes_dict = val_classes_dict_r

        self.test_data_dict_r = test_data_dict_r
        self.test_labels_dict_r = test_labels_dict_r
        self.test_classes_dict_r = test_classes_dict_r

        

class MLBF_Results:
    #### Class to hold the results from a series of test runs

    def __init__(self, test_name="", meas_combo_labels=[], max_num_meas=12, results_dir='./Results'):
        # Overall result configs
        self.results_dir = results_dir
        self.test_name = test_name
        self.meas_combo_labels = meas_combo_labels

        # Setup Pandas parameters to save the results at the end
        #   Setting idx:|--0-------|--1-----|--2-------|--3--------|--4------------|--5----------|--6--------|--7---------------|--8--------------|   
        self.all_cols = ["num_meas", "model", "channel", "test_acc", "confusionMat", "pe_byLabel", "gainLoss", "codebook_labels", "codebook_beams"]
        self.all_data = []  # Temporary list to hold the data (append lists with the above information)
        self.all_df   = []  # Later will be filled with the Pandas DataFrame


    def save_results(self, num_meas, model_label, channel_tag,  
                     all_curMeas_acc, all_curMeas_test_confusion, 
                     all_curMeas_pe, all_gainloss_perc, codebook_labels, all_use_beams):
        ## Save results, indexed by:
        #   Col 0: Number of measurements
        #   Col 1: Model label (ex: CNN, FCnet, Agile-link)
        #   Col 2: Channel label (basic info about the channel used; ex: alpha)
        #   Col 3-6: Actual data/results
        data_row = [num_meas, model_label, channel_tag, all_curMeas_acc, all_curMeas_test_confusion, all_curMeas_pe, all_gainloss_perc, codebook_labels, all_use_beams]
        self.all_data.append(data_row)


    def copy_results(self, mlbf_results_i, entry_idxs=[]):
        ## Save the results from another results structure into this results structure
        num_results = len(mlbf_results_i.all_data)

        # Check which tags should be used
        if (len(entry_idxs) == 0):
            entry_idxs = range(num_results)

        # Loop through all the results in the donating structure
        for result_i in entry_idxs:
            self.all_data.append(mlbf_results_i.all_data[result_i])


    def pkl_save_results(self):
        ## Save total results to a pickle file
        with open(os.path.join(self.results_dir, 'all_{}.pkl'.format(self.test_name)), 'wb') as f:
            pickle.dump([self.meas_combo_labels, self.all_data], f)


    def pkl_load_results(self):
        ## Load total results from a pickle file
        with open(os.path.join(self.results_dir, 'all_{}.pkl'.format(self.test_name)), 'rb') as f:
            self.meas_combo_labels, self.all_data = pickle.load(f)


    def plot_results(self, plot_metric=1, plot_type=1, 
                     test_title="", sim_exp="exp",
                     gain_loss_perc=90,
                     fixed_model="CNN", fixed_channel_rel_pwr_sim=17.4, fixed_channel_rel_pwr_exp=20, fixed_channel_snr=15, fixed_meas_combo="1 PN Beams, [0-9] SA Beams",
                     max_gain_loss=2.0, bar_nlos_rel_pwrs=[],
                     SAVE_PLOTS=False, plot_width=-1, plot_height=-1, save_type="png", plot_tight_layout=False):
        ## Plot the results and save the plots to files
        #  Plot metrics:
        #   1. Gain loss (DFT gain for the best beam - gain for the selected beam)
        #   2. Test accuracy
        #   3. Required number of measurements vs. alpha (used with plot types 1)
        #  Plot types: (plot metric vs number of measurements unless otherwise listed)
        #   1. Comparison of models (fixed channel characteristics and measurement combo)
        #   2. Comparison of measurement combos (fixed channel characteristics and model)
        #   3. Comparison of channel performance (fixed measurement combo and model)

        # Constants
        include_allSNR_results = False
        include_allMeasCombos_plot3 = True

        if not include_allMeasCombos_plot3:
            all_models = [cnn, fcnet, csmp] 
            all_models_labels = ["Proposed CNN", "mmRAPID", "RSS-MP"] 
            all_models_meas = ["3 PN Beams, [0-9]* SA Beams", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z"] 
            all_models_meas_labels = ["3 PN, SA Beams", "PN Beams", "(PN Beams)", "(PN Beams)"]
            #all_models_meas = ["[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z"] 
            #all_models_meas = ["\A[0-9]* SA Beams", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z"] 
        else:
            # Used for #3 (line graph)
            # all_models = [cnn, cnn, cnn, cnn, fcnet, csmp] 
            # all_models_labels = ["Proposed CNN", "Proposed CNN", "Proposed CNN", "Proposed CNN", "mmRAPID", "RSS-MP"] 
            # all_models_meas = ["1 PN Beams, [0-9]* SA Beams", "2 PN Beams, [0-9]* SA Beams", "3 PN Beams, [0-9]* SA Beams", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z"] 
            # Used for #4 (bar graph)
            all_models = [cnn, cnn, cnn, cnn, fcnet, csmp] 
            all_models_labels = ["CNN", "CNN", "CNN", "CNN", "mmRAPID", "RSS-MP"] 
            all_models_meas = ["1 PN Beams, [0-9]* SA Beams", "2 PN Beams, [0-9]* SA Beams", "3 PN Beams, [0-9]* SA Beams", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z"] 
            all_models_meas_labels = ["1 PN, SA Beams", "2 PN, SA Beams", "3 PN, SA Beams", "PN Beams", "PN Beams", "PN Beams"]
            # all_models = [cnn, cnn, cnn, fcnet, csmp] 
            # all_models_labels = ["Proposed CNN", "Proposed CNN", "Proposed CNN", "mmRAPID", "RSS-MP"] 
            # all_models_meas = ["2 PN Beams, [0-9]* SA Beams", "3 PN Beams, [0-9]* SA Beams", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z", "[0-9]* PN Beams\Z"] 

        # all_models = [cnn, fcnet, agilelink, csmp] 
        # all_models_labels = ["Proposed CNN", "mmRAPID", "RSS-MP", "RSS-MP"]
        # all_models_meas = ["1 PN Beams, [0-9]* SA Beams", "[0-9]* PN Beams\Z", "\A[0-9]* SA Beams", "[0-9]* PN Beams\Z"]
        
        all_num_meas = [4, 5, 6, 7, 8, 9, 10]
        # all_beam_combos = ["[0-9] PN Beams\Z", "1 PN Beams, [0-9] SA Beams", "2 PN Beams, [0-9] SA Beams", "3 PN Beams, [0-9] SA Beams", "\A[0-9] SA Beams", "\A[0-9] QPD Beams", "1 PN Beams, [0-9] QPD Beams"]
        all_beam_combos = ["\A[0-9]* SA Beams", "1 PN Beams, [0-9]* SA Beams", "2 PN Beams, [0-9]* SA Beams", "3 PN Beams, [0-9]* SA Beams", "[0-9]* PN Beams\Z", "1 PN Beams, [0-9]* QPD Beams"]
        all_channel_nlos_sim = [17.4, 13.0, 8.69, 6.51, 4.34, 3.00, 2.17, 1.09, 0.52]
        # all_channel_nlos_exp = [20, 10, 10, 5, 6.2, 3.6, 1, 0.5]
        all_channel_nlos_exp = [20, 10, 10, 5, 3.6, 1, 0.5]

        # Derived Constants
        included_num_meas = np.array(all_num_meas)
        resize_plots = False
        if (plot_width != -1) and (plot_height != -1):
            resize_plots = True

        # Default test title
        if (test_title == ""):
            test_title = self.test_name

        # Update the Pandas Dataframe copy for easier plotting
        self.all_df = pd.DataFrame(self.all_data)
        self.all_df.columns = self.all_cols
        
        # Unpack the results into a new dataframe for easier plotting
        num_points = len(self.all_df.index)
        exp_cols = ["num_meas", "model", "channel", "test_acc", "confusionMat", "pe_byLabel", "gainLoss", "codebook_labels", "codebook_beams", "channel_rel_pwr", "channel_snr", "sim_or_exp"]
        expanded_data = []
        for pt_i in np.arange(num_points):
            # Data used for all sub-points in a point
            num_meas_i = self.all_df["num_meas"][pt_i]
            model_i = self.all_df["model"][pt_i]
            channel_i = self.all_df["channel"][pt_i]
            codebook_labels_i = self.all_df["codebook_labels"][pt_i]
            codebook_beams_i = self.all_df["codebook_beams"][pt_i]

            # Data that contains different data per sub-point
            test_acc_i_all = self.all_df["test_acc"][pt_i]              # (# beam combos) x (# subpoints)
            confusionMat_i_all = self.all_df["confusionMat"][pt_i]      # (# beam combos) x (# subpoints) x (# of labels used) x (# of labels used)
            pe_byLabel_i_all = self.all_df["pe_byLabel"][pt_i]          # (# beam combos) x (# subpoints) x (# of labels used)
            gainLoss_i_all = self.all_df["gainLoss"][pt_i]              # (# subpoints) x (# beam combos) x (# percentiles = 11)

            # Loop through all the subpoints (each subpoint represents a different channel, typically SNR and/or alpha value)
            num_subpoints = gainLoss_i_all.shape[0]
            num_subpoints_include = num_subpoints
            if not include_allSNR_results:
                num_subpoints_include = num_subpoints_include - 1
            for spt_j in np.arange(num_subpoints_include):
                # Extract the channel info
                if spt_j < num_subpoints-1:
                    channel_i_rel_pwr = channel_i.NLOS_PATH_REL_PWR[spt_j]
                    channel_i_snr = channel_i.DATA_SNR[spt_j]
                    data_type_i = channel_i.sim_or_exp[spt_j]
                else:
                    channel_i_rel_pwr = "ALL"
                    channel_i_snr = "ALL"
                    data_type_i = "ALL"

                # Loop through all the beam combos
                num_beam_combos = len(codebook_labels_i)
                for beam_i in np.arange(num_beam_combos):
                    # Store the data into a new list
                    codebook_beams_i_k = {"PN": codebook_beams_i["PN"][beam_i, :], "QPD": codebook_beams_i["QPD"][beam_i, :], "DFTSA": codebook_beams_i["DFTSA"][beam_i, :]}
                    data_pt_i_j_k = [num_meas_i, model_i, channel_i, \
                                    test_acc_i_all[beam_i, spt_j], confusionMat_i_all[beam_i,spt_j,:,:], pe_byLabel_i_all[beam_i,spt_j,:], gainLoss_i_all[spt_j,beam_i,:], \
                                    codebook_labels_i[beam_i], codebook_beams_i_k, \
                                    channel_i_rel_pwr, channel_i_snr, data_type_i]
                    # data_pt_i_j = [num_meas_i, model_i, channel_i, \
                    #             test_acc_i_all[:, spt_j], confusionMat_i_all[:,spt_j,:,:], pe_byLabel_i_all[:,spt_j,:], gainLoss_i_all[spt_j,:,:], \
                    #             codebook_labels_i, codebook_beams_i, \
                    #             channel_i_rel_pwr, channel_i_snr]
                    expanded_data.append(data_pt_i_j_k)

        # Export to a dataframe
        expanded_df = pd.DataFrame(expanded_data)
        expanded_df.columns = exp_cols

        # Plot the results
        #   - Meas combos:      self.all_df.loc[self.all_df['model'].str.contains(agilelink)]["codebook_labels"]                (for labels)
        #                       self.all_df.loc[self.all_df['model'].str.contains(agilelink)]["codebook_beams"][idx][beam_type] (for beam indices)
        #   - SNRs:             self.all_df.loc[self.all_df['model'].str.contains(agilelink)]["channel"][idx].DATA_SNR
        #   - Rel path pwrs:    self.all_df.loc[self.all_df['model'].str.contains(agilelink)]["channel"][idx].NLOS_PATH_REL_PWR
        #   - Gain loss:        self.all_df.loc[self.all_df['model'].str.contains(agilelink)]["gainLoss"]
        #   - Results with a specific range of rel pwrs: 
        #                       expanded_df.loc[(expanded_df['channel_rel_pwr'] < 6)].loc[(expanded_df['channel_rel_pwr'] > 3)]['codebook_labels']
        gain_loss_idx = int(gain_loss_perc/10)

        if plot_metric == 1:
            if plot_type == 1:
                ##### 1.1 Gain loss for different models vs number of measurements (with a fixed channel and measurement combos)
                fixed_alpha_sim = -np.log(10**(-fixed_channel_rel_pwr_sim/20))
                fixed_alpha_exp = -np.log(10**(-fixed_channel_rel_pwr_exp/20))

                # Setup the figure options
                fig = plt.figure()
                rc('text', usetex=True)
                sim_or_exp_plot = ["sim", "exp"]
                sim_or_exp_labels = ["Sim. Data", "Exp. Data"]
                if ('QPD' not in fixed_meas_combo) and ('SA' not in fixed_meas_combo):
                    models_plot = [fcnet, cnn, csmp]
                    models_plot_labels = ["MLP", "CNN", "CS-RSS-MP"]
                    all_num_meas = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
                    included_num_meas = np.array(all_num_meas)
                    print("\tUsing all PN models...")
                elif ('QPD' not in fixed_meas_combo) and ('PN' not in fixed_meas_combo):
                    models_plot = [fcnet, cnn, agilelink]
                    models_plot_labels = ["MLP", "CNN", "SA-RSS-MP"]
                    print("\tUsing all multifinger models...")
                else:
                    models_plot = [fcnet, cnn]
                    models_plot_labels = ["MLP", "CNN"]
                    print("\tUsing only ML models...")

                fixed_meas_combo_label = fixed_meas_combo
                if "\Z" in fixed_meas_combo_label:
                    fixed_meas_combo_label = fixed_meas_combo_label.replace('\Z', '')
                if "\A" in fixed_meas_combo_label:
                    fixed_meas_combo_label = fixed_meas_combo_label.replace('\A', '')
                fixed_meas_combo_label = fixed_meas_combo_label.replace('[0-9]*', '')
                fixed_meas_combo_label = fixed_meas_combo_label.replace('1 PN Beams', '1 PN Beam')

                # colors_sim = [(65/255, 105/255, 225/255), (34/255, 139/255, 34/255)]
                # colors_exp = [(0/255, 0/255, 139/255), (0/255, 100/255, 0/255)]
                colors_sim = [(65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (204/255, 102/255, 0/255)]
                colors_exp = [(65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (204/255, 102/255, 0/255)]
                colors_plot = [colors_sim, colors_exp]
                line_sim = ['--', '--', '--', '--']
                line_exp = ['-', '-', '-', '-']
                line_plot = [line_sim, line_exp]
                style_sim = ['b--', 'g--', 'r--', 'm--']
                style_exp = ['b-', 'g-', 'r-', 'm-']
                style_plot = [style_sim, style_exp]

                for data_plot_i in np.arange(len(sim_or_exp_plot)):
                    # Setup for the right data type
                    if sim_or_exp_plot[data_plot_i] == "sim":
                        fixed_channel_rel_pwr = fixed_channel_rel_pwr_sim
                    else:
                        fixed_channel_rel_pwr = fixed_channel_rel_pwr_exp

                    # Find the data with the right channel setup
                    eligible_data_chan_allSNR = expanded_df.loc[expanded_df['channel_rel_pwr'] == fixed_channel_rel_pwr]
                    eligible_data_chan = eligible_data_chan_allSNR.loc[eligible_data_chan_allSNR['channel_snr'] == fixed_channel_snr]
                    # Find the data with the right type (experiment or simulation)
                    eligible_data_type = eligible_data_chan.loc[eligible_data_chan['sim_or_exp'] == sim_or_exp_plot[data_plot_i]]

                    # Look through the requested models
                    for model_plot_i in np.arange(len(models_plot)):
                        
                        # Find the data with the right model used (CNN or FCnet)
                        eligible_data_model = eligible_data_type.loc[eligible_data_type['model'] == models_plot[model_plot_i]]
                        # Filter out the data with the desired codebook
                        combo_data = eligible_data_model.loc[eligible_data_model['codebook_labels'].str.contains(fixed_meas_combo)]

                        #meas_combos_plot = eligible_data["codebook_labels"].unique()
                        #models_plot = eligible_data["model"].unique()

                        # Extract the final data used for the x and y axes (gain loss vs number of measurements)
                        gainLoss_plot = np.array(combo_data["gainLoss"].tolist())[:,gain_loss_idx]
                        gainLoss_meas = np.array(combo_data["num_meas"].tolist())
                        #import pdb; pdb.set_trace()
                        # Find the alpha labels for the plot title
                        fixed_alpha_sim = -np.log(10**(-fixed_channel_rel_pwr_sim/20))
                        fixed_alpha_exp = -np.log(10**(-fixed_channel_rel_pwr_exp/20))

                        # plt.plot(gainLoss_meas, gainLoss_plot, style_plot[data_plot_i][model_plot_i], 
                        #         marker="o", lw=4, label="{}, {}".format(models_plot_labels[model_plot_i], sim_or_exp_labels[data_plot_i]))
                        plt.plot(np.sort(gainLoss_meas), gainLoss_plot[np.argsort(gainLoss_meas)], line_plot[data_plot_i][model_plot_i], color=colors_plot[data_plot_i][model_plot_i], 
                                marker="o", lw=3, label="{}, {}".format(models_plot_labels[model_plot_i], sim_or_exp_labels[data_plot_i]))
                if resize_plots:
                    plt.legend(loc="best")
                else:
                    plt.legend(loc="upper right")
                plt.title("Model Comparison: {} \n".format(fixed_meas_combo_label) + r"(Sim. Channel $\alpha={:.2f}$, Exp. Channel $\alpha={:.2f}$)".format(fixed_alpha_sim, fixed_alpha_exp))
                plt.xlabel("Number of Measurements (M)")
                # plt.ylabel("{}th Percentile Post-Alignment Gain Loss (dB)".format(GAIN_LOSS_PERCENTILES[gain_loss_idx]))
                plt.ylabel("{}th Percentile Gain Loss (dB)".format(GAIN_LOSS_PERCENTILES[gain_loss_idx]))
                plt.grid(True)
                plt.xlim(np.min(included_num_meas),np.max(included_num_meas))
                if resize_plots:
                    # fig.set_size_inches(6, 2.5)
                    fig.set_size_inches(plot_width, plot_height)
                    fig.subplots_adjust(top=(fig.subplotpars.top - 0.05))
                    fig.subplots_adjust(bottom=0.17)
                plt.draw()
                if SAVE_PLOTS:
                    if resize_plots:
                        plt.savefig(os.path.join(self.results_dir, "{}_small.{}".format(test_title, save_type)))
                    else:
                        plt.savefig(os.path.join(self.results_dir, "{}.{}".format(test_title, save_type)))


            elif plot_type == 2:
                #### 1.2. Comparison of measurement combos (fixed channel characteristics and model)

                # Setup the figure options
                fig = plt.figure()
                rc('text', usetex=True)

                sim_or_exp_plot = [sim_exp]
                if sim_exp == "sim":
                    sim_or_exp_labels = ["Sim. Data"]
                elif sim_exp == "exp":
                    sim_or_exp_labels = ["Exp. Data"]

                # Colors v1: Indigo, Purple, Dark Blue, Blue, Green, Burgundy, Crimson, Orange
                # colors_sim = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                # colors_exp = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                
                # Colors v2: Purple, Dark Blue, Blue, Green, Dark Green, Burgundy, Crimson, Orange
                #   Light Purple: (150/255, 0/255, 205/255)
                #   Light Red/Crimson?: (220/255, 20/255, 60/255)
                # colors_sim = [(112/255, 39/255, 195/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (53/255, 98/255, 68/255), (128/255, 0/255, 0/255), (202/255, 0/255, 42/255), (204/255, 102/255, 0/255)]
                # colors_exp = [(112/255, 39/255, 195/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (53/255, 98/255, 68/255), (128/255, 0/255, 0/255), (202/255, 0/255, 42/255), (204/255, 102/255, 0/255)]

                # Colors v3: Purple, Blue, Blue, Blue, Green, Burgundy
                # colors_sim = [(112/255, 39/255, 195/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                # colors_exp = [(112/255, 39/255, 195/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                
                # Colors v4: Indigo, Blue, Blue, Blue, Green, Burgundy
                colors_sim = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                colors_exp = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]

                colors_plot = [colors_sim, colors_exp]

                # # Same line type for all sim/exp
                # line_sim = ['--', '--', '--', '--', '--', '--', '--', '--']
                # line_exp = ['-', '-', '-', '-', '-', '-', '-', '-']

                # Line types v3: Different line types with same color for the combinations of PN beams and SA beams
                line_sim = ['--', '--', '--', '--', '--', '--', '--', '--']
                line_exp = ['-', '-', ':', '--', '-', '-', '-', '-']

                line_plot = [line_sim, line_exp]
                
                style_sim = ['b--', 'g--', 'r--', 'm--']
                style_exp = ['b-', 'g-', 'r-', 'm-']
                style_plot = [style_sim, style_exp]

                for data_plot_i in np.arange(len(sim_or_exp_plot)):
                    # Setup for the right data type
                    if sim_or_exp_plot[data_plot_i] == "sim":
                        fixed_channel_rel_pwr = fixed_channel_rel_pwr_sim
                        type_i = 0
                    else:
                        fixed_channel_rel_pwr = fixed_channel_rel_pwr_exp
                        type_i = 1

                    # Find the data with the right channel setup
                    eligible_data_chan_allSNR = expanded_df.loc[expanded_df['channel_rel_pwr'] == fixed_channel_rel_pwr]
                    eligible_data_chan = eligible_data_chan_allSNR.loc[eligible_data_chan_allSNR['channel_snr'] == fixed_channel_snr]
                    # Find the data with the right type (experiment or simulation)
                    eligible_data_type = eligible_data_chan.loc[eligible_data_chan['sim_or_exp'] == sim_or_exp_plot[data_plot_i]]

                    # Look through the requested models
                    all_beam_combos_labels = []
                    for meas_plot_i in np.arange(len(all_beam_combos)):
                        
                        # Create the label for the current measurement combo
                        meas_combo = all_beam_combos[meas_plot_i]
                        meas_combo_label = meas_combo
                        if "\Z" in meas_combo_label:
                            meas_combo_label = meas_combo_label.replace('\Z', '')
                        if "\A" in meas_combo_label:
                            meas_combo_label = meas_combo_label.replace('\A', '')
                        meas_combo_label = meas_combo_label.replace('[0-9]*', '')
                        meas_combo_label = meas_combo_label.replace('1 PN Beams', '1 PN Beam')
                        all_beam_combos_labels.append(meas_combo_label)

                        # Find the data with the right model used
                        eligible_data_model = eligible_data_type.loc[eligible_data_type['model'] == fixed_model]
                        # Filter out the data with the desired codebook
                        combo_data = eligible_data_model.loc[eligible_data_model['codebook_labels'].str.contains(meas_combo)]

                        # Extract the final data used for the x and y axes (gain loss vs number of measurements)
                        if fixed_model == fcnet:
                            import pdb; pdb.set_trace()
                        gainLoss_plot = np.array(combo_data["gainLoss"].tolist())[:,gain_loss_idx]
                        gainLoss_meas = np.array(combo_data["num_meas"].tolist())
                        #import pdb; pdb.set_trace()
                        # Find the alpha labels for the plot title
                        fixed_alpha_sim = -np.log(10**(-fixed_channel_rel_pwr_sim/20))
                        fixed_alpha_exp = -np.log(10**(-fixed_channel_rel_pwr_exp/20))

                        # plt.plot(gainLoss_meas, gainLoss_plot, style_plot[type_i][meas_plot_i], 
                        #         marker="o", lw=4, label="{}, {}".format(all_beam_combos[meas_plot_i], sim_or_exp_labels[type_i]))
                        line_format_i = type_i
                        if (len(sim_or_exp_plot) == 1):
                            line_format_i = 1
                        plt.plot(np.sort(gainLoss_meas), gainLoss_plot[np.argsort(gainLoss_meas)], line_plot[line_format_i][meas_plot_i], color=colors_plot[type_i][meas_plot_i], 
                                marker="o", lw=3, label="{}".format(all_beam_combos_labels[meas_plot_i]))
                sim_or_exp_label = r"(Sim. Channel $\alpha={:.2f}$, Exp. Channel $\alpha={:.2f}$)".format(fixed_alpha_sim, fixed_alpha_exp)
                if (len(sim_or_exp_plot) == 1) and (sim_or_exp_plot[0] == 'sim'):
                    sim_or_exp_label = r"(Sim. Channel $\alpha={:.2f}$)".format(fixed_alpha_sim)
                elif (len(sim_or_exp_plot) == 1) and (sim_or_exp_plot[0] == 'exp'):
                    sim_or_exp_label = r"(Exp. Channel $\alpha={:.2f}$)".format(fixed_alpha_exp)
                if resize_plots:
                    plt.legend(loc="best")
                else:
                    plt.legend(loc="upper right")
                plt.title("Feature Comparison: {} Algorithm with {}\n".format(fixed_model, sim_or_exp_labels[0]) + sim_or_exp_label)
                plt.xlabel("Number of Measurements (M)")
                # plt.ylabel("{}th Percentile Post-Alignment Gain Loss (dB)".format(GAIN_LOSS_PERCENTILES[gain_loss_idx]))
                plt.ylabel("{}th Percentile Gain Loss (dB)".format(GAIN_LOSS_PERCENTILES[gain_loss_idx]))
                plt.grid(True)
                plt.xlim(np.min(included_num_meas),np.max(included_num_meas))
                if resize_plots:
                    # fig.set_size_inches(6, 2.5)
                    fig.set_size_inches(plot_width, plot_height)
                    fig.subplots_adjust(top=(fig.subplotpars.top - 0.05))
                    fig.subplots_adjust(bottom=0.17)
                plt.draw()
                if SAVE_PLOTS:
                    if resize_plots:
                        plt.savefig(os.path.join(self.results_dir, "{}_small.{}".format(test_title, save_type)))
                    else:
                        plt.savefig(os.path.join(self.results_dir, "{}.{}".format(test_title, save_type)))


            elif plot_type == 3:
                #### 1.3. Comparison of channel performance - gain loss vs channel alpha

                # # Function parameters
                # # sim_exp="exp",
                # # fixed_channel_snr=15, 
                # # max_gain_loss=2.0,

                # # Setup the figure options
                # fig = plt.figure()
                # rc('text', usetex=True)

                # # if pn_combo_avg == -1:
                # #     plt.plot(included_num_meas, plot_all_gainLoss[:, len(DATA_SNR), percentile_indices], marker=".")
                # #     #plt.legend(("{}th percentile".format(GAIN_LOSS_PERCENTILES[percentile_indices[0]]), "{}th percentile".format(GAIN_LOSS_PERCENTILES[percentile_indices[1]])))
                # # else:
                # #     for file_i in np.arange(num_files):
                # #         plt.plot(included_num_meas, plot_all_gainLoss[file_i, pn_combo_idxs[file_i], :, snr_idx, 9], 
                # #                 marker="o", lw=4, label=label_all[file_i])
                # #     plt.legend(loc="upper right")

                # # Create the label for the current data type (sim or exp)
                # sim_or_exp_plot = [sim_exp]
                # if sim_exp == "sim":
                #     sim_or_exp_labels = ["Sim."]
                #     all_channel_nlos = all_channel_nlos_sim
                # elif sim_exp == "exp":
                #     sim_or_exp_labels = ["Exp."]
                #     all_channel_nlos = all_channel_nlos_exp
                
                # # Find the alpha labels for the plot title
                # all_channel_alpha = -np.log(10**(-np.array(all_channel_nlos)/20))

                # # # Indigo, Purple, Dark Blue, Blue, Green, Burgundy, Crimson, Orange
                # # colors_sim = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                # # colors_exp = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                
                # # Indigo, Blue, Green, Burgundy, Crimson, Orange
                # colors_sim = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                # colors_exp = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                # colors_plot = [colors_sim, colors_exp]
                # line_sim = ['--', '--', '--', '--', '--', '--', '--', '--']
                # line_exp = ['-', '-', '-', '-', '-', '-', '-', '-']
                # line_plot = [line_sim, line_exp]
                # style_sim = ['b--', 'g--', 'r--', 'm--']
                # style_exp = ['b-', 'g-', 'r-', 'm-']
                # style_plot = [style_sim, style_exp]
                
                # for data_plot_i in np.arange(len(sim_or_exp_plot)):
                #     # Setup for the right data type
                #     if sim_or_exp_plot[data_plot_i] == "sim":
                #         type_i = 0
                #     else:
                #         type_i = 1

                    
                #     # Filter out the data with the desired SNR
                #     eligible_data_chan = expanded_df.loc[expanded_df['channel_snr'] == fixed_channel_snr]
                #     # Find the data with the right type (experiment or simulation)
                #     eligible_data_type = eligible_data_chan.loc[eligible_data_chan['sim_or_exp'] == sim_or_exp_plot[data_plot_i]]

                #     # Look through the requested models
                #     for model_i in np.arange(len(all_models)):
                        
                #         # Create the label for the current measurement combo
                #         meas_combo = all_models_meas[model_i]
                #         meas_combo_label = meas_combo
                #         if "\Z" in meas_combo_label:
                #             meas_combo_label = meas_combo_label.replace('\Z', '')
                #         if "\A" in meas_combo_label:
                #             meas_combo_label = meas_combo_label.replace('\A', '')
                #         meas_combo_label = meas_combo_label.replace('[0-9]*', '')
                #         meas_combo_label = meas_combo_label.replace('1 PN Beams', '1 PN Beam')
                #         print("\t\tTesting model: {}, {}".format(all_models_labels[model_i], meas_combo_label))

                #         # Filter out the data with the desired codebook
                #         eligible_data_meas = eligible_data_type.loc[eligible_data_type['codebook_labels'].str.contains(meas_combo)]
                        
                #         # Find the data with the right model used
                #         eligible_data_model = eligible_data_meas.loc[eligible_data_meas['model'] == all_models[model_i]]
                    
                #         # Initialize the number of required measurements array
                #         num_required_meas_i = default_num_meas * np.ones(all_channel_alpha.shape)

                #         # Compute the required number of measurements
                #         for chan_plot_i in np.arange(len(all_channel_alpha)):

                #             # Find the data with the right channel setup
                #             combo_data = eligible_data_model.loc[eligible_data_model['channel_rel_pwr'] == all_channel_nlos[chan_plot_i]]

                #             # Find the gain loss for each number of measurements
                #             gainLoss_plot = np.array(combo_data["gainLoss"].tolist())[:,gain_loss_idx]
                #             gainLoss_meas = np.array(combo_data["num_meas"].tolist())

                #             # Sort the data (may not be in order)
                #             # gainLoss_ordered = np.flip(np.sort(gainLoss_plot))
                #             # gainLoss_ord_meas = gainLoss_meas[np.flip(np.argsort(gainLoss_plot))]
                #             gainLoss_ordered = gainLoss_plot[np.argsort(gainLoss_meas)]
                #             gainLoss_ord_meas = np.sort(gainLoss_meas)

                #             # Compute the number of required measurements for this channel/alpha
                #             all_within_loss = (gainLoss_ordered < max_gain_loss)
                #             num_within_loss = np.sum(all_within_loss)
                #             if num_within_loss <= 0:
                #                 continue
                #             else:
                #                 num_required_meas_i[chan_plot_i] = gainLoss_ord_meas[np.argmax(all_within_loss)]
                #             #import pdb; pdb.set_trace()

                #         # plt.plot(gainLoss_meas, gainLoss_plot, style_plot[type_i][model_i], 
                #         #         marker="o", lw=4, label="{}, {}".format(all_beam_combos[model_i], sim_or_exp_labels[type_i]))
                #         line_format_i = type_i
                #         if (len(sim_or_exp_plot) == 1):
                #             line_format_i = 1
                #         # Plot the results
                #         #import pdb; pdb.set_trace()
                #         plt.plot(np.sort(all_channel_alpha), num_required_meas_i[np.argsort(all_channel_alpha)], line_plot[line_format_i][model_i], color=colors_plot[type_i][model_i], 
                #                 marker="o", lw=4, label="{}, {}".format(all_models_labels[model_i], meas_combo_label))
                # plt.title("Number of Required Measurements for {} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                # plt.xlabel(r"{} Channel Multipath Strength ($\alpha$)".format(sim_or_exp_labels[0]))
                # plt.ylabel("Number of Required Meas. (M)")
                # plt.grid(True)
                # # plt.ylim(np.min(included_num_meas),np.max(included_num_meas))
                # if sim_exp == 'sim':
                #     plt.legend(loc="upper right")
                #     plt.ylim(4, 12)
                # else:
                #     plt.legend(loc="upper right")
                #     plt.ylim(4,default_num_meas+0.1)
                # plt.xlim(np.min(all_channel_alpha),np.max(all_channel_alpha))
                # if resize_plots:
                #     # fig.set_size_inches(6, 2.5)
                #     fig.set_size_inches(plot_width, plot_height)
                #     fig.subplots_adjust(top=(fig.subplotpars.top - 0.05))
                #     fig.subplots_adjust(bottom=0.17)
                # plt.draw()
                # if SAVE_PLOTS:
                #     if resize_plots:
                #         plt.savefig(os.path.join(self.results_dir, "{}_small.{}".format(test_title, save_type)))
                #     else:
                #         plt.savefig(os.path.join(self.results_dir, "{}.{}".format(test_title, save_type)))

                pass

            else:
                print("ERROR: INVALID PLOT TYPE \"{}\"".format(plot_type))


        elif plot_metric == 2:
            pass


        elif plot_metric == 3:
            if plot_type == 1:
                # Required # of measurments vs channel alpha for different models (with fixed measurement combos)
                #fixed_channel_rel_pwr 
                pass


            elif plot_type == 2:
                #### 3.2. Required # of measurments vs channel alpha for different channels (with fixed measurement combos and model)
                ####      Basically 3.3. but BAR GRAPH FORM

                # Function parameters
                alpha_width = 0.8
                label_format = "{:.2f}" #r"\alpha = {}"

                # Computed function parameters
                bar_all_locs = []                                                           # Bar coordinates for each model - list (by model) of lists (by alpha)
                bar_all_reqMeas = []                                                        # Bar heights for each model     - list (by model) of lists (by alpha)
                bar_alphas = -np.log(10**(-np.array(bar_nlos_rel_pwrs)/20))                 # Alpha values computed from bar_nlos_rel_pwrs
                bar_labels = [label_format.format(item) for item in bar_alphas] # Labels used for each set of bars (i.e. alpha labels)
                num_bars = len(all_models)
                bar_width = alpha_width/num_bars
                num_alphas = len(bar_nlos_rel_pwrs)
                bar_idxs = np.arange(num_alphas)
                all_bars = []

                # Setup the figure options
                #fig = plt.figure()
                fig, ax = plt.subplots()
                rc('text', usetex=True)
                ax.grid(zorder=0)
                ax.set_axisbelow(True)

                # if pn_combo_avg == -1:
                #     plt.plot(included_num_meas, plot_all_gainLoss[:, len(DATA_SNR), percentile_indices], marker=".")
                #     #plt.legend(("{}th percentile".format(GAIN_LOSS_PERCENTILES[percentile_indices[0]]), "{}th percentile".format(GAIN_LOSS_PERCENTILES[percentile_indices[1]])))
                # else:
                #     for file_i in np.arange(num_files):
                #         plt.plot(included_num_meas, plot_all_gainLoss[file_i, pn_combo_idxs[file_i], :, snr_idx, 9], 
                #                 marker="o", lw=4, label=label_all[file_i])
                #     plt.legend(loc="upper right")

                # Create the label for the current data type (sim or exp)
                sim_or_exp_plot = [sim_exp]
                if sim_exp == "sim":
                    sim_or_exp_labels = ["Sim."]
                elif sim_exp == "exp":
                    sim_or_exp_labels = ["Exp."]

                # # Indigo, Purple, Dark Blue, Blue, Green, Burgundy, Crimson, Orange
                # colors_sim = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                # colors_exp = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                
                if not include_allMeasCombos_plot3:
                    # Indigo, Blue, Green, Burgundy, Crimson, Orange
                    colors_sim = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                    colors_exp = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                    colors_plot = [colors_sim, colors_exp]
                    hatch_sim = ['', '', '', '', '', '', '', '']
                    hatch_exp = ['', '', '', '', '', '', '', '']
                    hatch_plot = [hatch_sim, hatch_exp]
                else:
                    # 1.25*Indigo, Indigo, 0.75*Indigo, Blue, Green, Burgundy
                    sc1 = 1.75
                    sc2 = 1
                    sc3 = 0.8
                    # colors_sim = [(sc1*75/255, sc1*0/255, sc1*130/255), (sc2*75/255, sc2*0/255, sc2*130/255), (sc3*75/255, sc3*0/255, sc3*130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    # colors_exp = [(sc1*75/255, sc1*0/255, sc1*130/255), (sc2*75/255, sc2*0/255, sc2*130/255), (sc3*75/255, sc3*0/255, sc3*130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    colors_sim = [(sc3*75/255, sc3*0/255, sc3*130/255), (sc1*75/255, sc1*0/255, sc1*130/255), (204/255, 102/255, 0/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    colors_exp = [(sc3*75/255, sc3*0/255, sc3*130/255), (sc1*75/255, sc1*0/255, sc1*130/255), (204/255, 102/255, 0/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    colors_plot = [colors_sim, colors_exp]
                    hatch_sim = ['', '', '', '', '', '', '', '']
                    hatch_exp = ['', '', '', '', '', '', '', '']
                    hatch_plot = [hatch_sim, hatch_exp]

                    # Indigo, Indigo, Blue, Green, Burgundy
                    # colors_sim = [(75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    # colors_exp = [(75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    # colors_plot = [colors_sim, colors_exp]
                    # # hatch_sim = ['', '/', '\\', '', '', '', '', '']
                    # # hatch_exp = ['', '/', '\\', '', '', '', '', '']
                    # hatch_sim = ['', '', '', '', '', '', '', '']
                    # hatch_exp = ['', '', '', '', '', '', '', '']
                    # hatch_plot = [hatch_sim, hatch_exp]
                
                default_num_meas = 50

                for data_plot_i in np.arange(len(sim_or_exp_plot)):
                    # Setup for the right data type
                    if sim_or_exp_plot[data_plot_i] == "sim":
                        type_i = 0
                    else:
                        type_i = 1
                    
                    # Filter out the data with the desired SNR
                    eligible_data_chan = expanded_df.loc[expanded_df['channel_snr'] == fixed_channel_snr]
                    # Find the data with the right type (experiment or simulation)
                    eligible_data_type = eligible_data_chan.loc[eligible_data_chan['sim_or_exp'] == sim_or_exp_plot[data_plot_i]]

                    # Look through the requested models
                    for model_i in np.arange(len(all_models)):
                        
                        # Create the label for the current measurement combo
                        meas_combo = all_models_meas[model_i]
                        meas_combo_label = meas_combo
                        if "\Z" in meas_combo_label:
                            meas_combo_label = meas_combo_label.replace('\Z', '')
                        if "\A" in meas_combo_label:
                            meas_combo_label = meas_combo_label.replace('\A', '')
                        meas_combo_label = meas_combo_label.replace('[0-9]*', '')
                        meas_combo_label = meas_combo_label.replace('1 PN Beams', '1 PN Beam')
                        print("\t\tTesting model: {}, {}".format(all_models_labels[model_i], meas_combo_label))

                        # Filter out the data with the desired codebook
                        eligible_data_meas = eligible_data_type.loc[eligible_data_type['codebook_labels'].str.contains(meas_combo)]
                        
                        # Find the data with the right model used
                        eligible_data_model = eligible_data_meas.loc[eligible_data_meas['model'] == all_models[model_i]]
                    
                        # Initialize the number of required measurements array
                        num_required_meas_i = default_num_meas * np.ones(bar_alphas.shape)

                        # Compute the required number of measurements
                        for chan_plot_i in np.arange(len(bar_alphas)):

                            # Find the data with the right channel setup
                            combo_data = eligible_data_model.loc[eligible_data_model['channel_rel_pwr'] == bar_nlos_rel_pwrs[chan_plot_i]]

                            # Find the gain loss for each number of measurements
                            gainLoss_plot = np.array(combo_data["gainLoss"].tolist())[:,gain_loss_idx]
                            gainLoss_meas = np.array(combo_data["num_meas"].tolist())

                            # Sort the data (may not be in order)
                            # gainLoss_ordered = np.flip(np.sort(gainLoss_plot))
                            # gainLoss_ord_meas = gainLoss_meas[np.flip(np.argsort(gainLoss_plot))]
                            gainLoss_ordered = gainLoss_plot[np.argsort(gainLoss_meas)]
                            gainLoss_ord_meas = np.sort(gainLoss_meas)

                            # Compute the number of required measurements for this channel/alpha
                            all_within_loss = (gainLoss_ordered < max_gain_loss)
                            num_within_loss = np.sum(all_within_loss)
                            # if all_models[model_i] == cnn:
                            #     print("{} channel, cnn: {} req. meas".format(bar_nlos_rel_pwrs[chan_plot_i], gainLoss_ord_meas[np.argmax(all_within_loss)]))
                            #     import pdb; pdb.set_trace()
                            
                            if num_within_loss <= 0:
                                continue
                            else:
                                num_required_meas_i[chan_plot_i] = gainLoss_ord_meas[np.argmax(all_within_loss)]

                        # Compute plot information
                        bar_loc_model_i = bar_idxs + bar_width*(model_i - ((num_bars-1)/2))
                        bar_all_locs.append(bar_loc_model_i)
                        bar_all_reqMeas.append(num_required_meas_i)

                        # import pdb; pdb.set_trace()
                        line_format_i = type_i
                        if (len(sim_or_exp_plot) == 1):
                            line_format_i = 1

                        # Plot model bars
                        bar_i = ax.bar(bar_loc_model_i, num_required_meas_i, bar_width, label="{}, {}".format(all_models_labels[model_i], all_models_meas_labels[model_i]), 
                                       color=colors_plot[type_i][model_i],
                                       linewidth=1, edgecolor='black', hatch=hatch_plot[line_format_i][model_i])
                        all_bars.append(bar_i)

                        # # Plot the results
                        # plt.plot(np.sort(all_channel_alpha), num_required_meas_i[np.argsort(all_channel_alpha)], line_plot[line_format_i][model_i], color=colors_plot[type_i][model_i], 
                        #         marker="o", lw=4, label="{}, {}".format(all_models_labels[model_i], meas_combo_label))
                
                if not resize_plots:
                    plt.title("Required Number of Measurements for {} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                plt.xlabel(r"{} Channel Multipath Strength ($\alpha$)".format(sim_or_exp_labels[0]))
                ax.set_xticks(bar_idxs)
                ax.set_xticklabels(bar_labels)
                plt.ylabel("Required Number of Meas. (M)")
                # plt.ylim(np.min(included_num_meas),np.max(included_num_meas))
                if sim_exp == 'sim':
                    if resize_plots:
                        plt.legend(loc="best")
                        # plt.title("Sim. Required Number of Measurements \nfor {} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                        plt.title("Sim. Required Number of Meas.\n{:.0f} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                    else:
                        plt.legend(loc="upper right")
                    # plt.ylim(4, default_num_meas+0.1)
                    plt.ylim(4, 36.1)
                else:
                    if resize_plots:
                        # plt.legend(loc="best")
                        # plt.title("Exp. Required Number of Measurements \nfor {} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                        plt.title("Exp. Required Number of Meas.\n{:.0f} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                    else:
                        plt.legend(loc="upper right")
                    # ax.ylim(4, default_num_meas+0.1)
                    plt.ylim(4, 36.1)
                # ax.xlim(np.min(all_channel_alpha),np.max(all_channel_alpha))
                # for model_i in np.arange(len(all_models)):      # Add labels on top of each bar - NOT SUPPORTED WITH THIS MATPLOTLIB
                #     ax.bar_label(all_bars[model_i], padding=3)  # SEE: https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
                # ax.set_yscale('log')

                if resize_plots:
                    # fig.set_size_inches(6, 2.5)
                    fig.set_size_inches(plot_width, plot_height)
                    fig.subplots_adjust(top=(fig.subplotpars.top - 0.05))
                    fig.subplots_adjust(bottom=0.17)
                    fig.tight_layout()
                plt.draw()
                if SAVE_PLOTS:
                    if resize_plots:
                        plt.savefig(os.path.join(self.results_dir, "{}_small.{}".format(test_title, save_type)))
                    else:
                        plt.savefig(os.path.join(self.results_dir, "{}.{}".format(test_title, save_type)))



            elif plot_type == 3:
                #### 3.3. Required # of measurments vs channel alpha for different channels (with fixed measurement combos and model)

                # Function parameters
                # sim_exp="exp",
                # fixed_channel_snr=15, 
                # max_gain_loss=2.0,

                # Setup the figure options
                fig = plt.figure()
                rc('text', usetex=True)

                # if pn_combo_avg == -1:
                #     plt.plot(included_num_meas, plot_all_gainLoss[:, len(DATA_SNR), percentile_indices], marker=".")
                #     #plt.legend(("{}th percentile".format(GAIN_LOSS_PERCENTILES[percentile_indices[0]]), "{}th percentile".format(GAIN_LOSS_PERCENTILES[percentile_indices[1]])))
                # else:
                #     for file_i in np.arange(num_files):
                #         plt.plot(included_num_meas, plot_all_gainLoss[file_i, pn_combo_idxs[file_i], :, snr_idx, 9], 
                #                 marker="o", lw=4, label=label_all[file_i])
                #     plt.legend(loc="upper right")

                # Create the label for the current data type (sim or exp)
                sim_or_exp_plot = [sim_exp]
                if sim_exp == "sim":
                    sim_or_exp_labels = ["Sim."]
                    all_channel_nlos = all_channel_nlos_sim
                elif sim_exp == "exp":
                    sim_or_exp_labels = ["Exp."]
                    all_channel_nlos = all_channel_nlos_exp
                
                # Find the alpha labels for the plot title
                all_channel_alpha = -np.log(10**(-np.array(all_channel_nlos)/20))

                # # Indigo, Purple, Dark Blue, Blue, Green, Burgundy, Crimson, Orange
                # colors_sim = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                # colors_exp = [(75/255, 0/255, 130/255), (84/255, 22/255, 180/255), (0/255, 0/255, 128/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                
                if not include_allMeasCombos_plot3:
                    # Indigo, Blue, Green, Burgundy, Crimson, Orange
                    colors_sim = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                    colors_exp = [(75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255), (220/255, 20/255, 60/255), (204/255, 102/255, 0/255)]
                    colors_plot = [colors_sim, colors_exp]
                    line_sim = ['--', '--', '--', '--', '--', '--', '--', '--']
                    line_exp = ['-', '-', '-', '-', '-', '-', '-', '-']
                    line_plot = [line_sim, line_exp]
                else:
                    # # Indigo, Indigo, Indigo, Blue, Green, Burgundy
                    # colors_sim = [(75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    # colors_exp = [(75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    # colors_plot = [colors_sim, colors_exp]
                    # line_sim = ['--', '--', '--', '--', '--', '--', '--', '--']
                    # line_exp = ['-', '--', ':', '-', '-', '-', '-', '-']
                    # line_plot = [line_sim, line_exp]

                    # Indigo, Indigo, Blue, Green, Burgundy
                    colors_sim = [(75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    colors_exp = [(75/255, 0/255, 130/255), (75/255, 0/255, 130/255), (65/255, 105/255, 225/255), (34/255, 139/255, 34/255), (128/255, 0/255, 0/255)]
                    colors_plot = [colors_sim, colors_exp]
                    line_sim = ['--', '--', '--', '--', '--', '--', '--', '--']
                    line_exp = ['-', '--', '-', '-', '-', '-', '-']
                    line_plot = [line_sim, line_exp]

                style_sim = ['b--', 'g--', 'r--', 'm--']
                style_exp = ['b-', 'g-', 'r-', 'm-']
                style_plot = [style_sim, style_exp]
                
                default_num_meas = 50

                for data_plot_i in np.arange(len(sim_or_exp_plot)):
                    # Setup for the right data type
                    if sim_or_exp_plot[data_plot_i] == "sim":
                        type_i = 0
                    else:
                        type_i = 1

                    
                    # Filter out the data with the desired SNR
                    eligible_data_chan = expanded_df.loc[expanded_df['channel_snr'] == fixed_channel_snr]
                    # Find the data with the right type (experiment or simulation)
                    eligible_data_type = eligible_data_chan.loc[eligible_data_chan['sim_or_exp'] == sim_or_exp_plot[data_plot_i]]

                    # Look through the requested models
                    for model_i in np.arange(len(all_models)):
                        
                        # Create the label for the current measurement combo
                        meas_combo = all_models_meas[model_i]
                        meas_combo_label = meas_combo
                        if "\Z" in meas_combo_label:
                            meas_combo_label = meas_combo_label.replace('\Z', '')
                        if "\A" in meas_combo_label:
                            meas_combo_label = meas_combo_label.replace('\A', '')
                        meas_combo_label = meas_combo_label.replace('[0-9]*', '')
                        meas_combo_label = meas_combo_label.replace('1 PN Beams', '1 PN Beam')
                        print("\t\tTesting model: {}, {}".format(all_models_labels[model_i], meas_combo_label))

                        # Filter out the data with the desired codebook
                        eligible_data_meas = eligible_data_type.loc[eligible_data_type['codebook_labels'].str.contains(meas_combo)]
                        
                        # Find the data with the right model used
                        eligible_data_model = eligible_data_meas.loc[eligible_data_meas['model'] == all_models[model_i]]
                    
                        # Initialize the number of required measurements array
                        num_required_meas_i = default_num_meas * np.ones(all_channel_alpha.shape)

                        # Compute the required number of measurements
                        for chan_plot_i in np.arange(len(all_channel_alpha)):

                            # Find the data with the right channel setup
                            combo_data = eligible_data_model.loc[eligible_data_model['channel_rel_pwr'] == all_channel_nlos[chan_plot_i]]

                            # Find the gain loss for each number of measurements
                            gainLoss_plot = np.array(combo_data["gainLoss"].tolist())[:,gain_loss_idx]
                            gainLoss_meas = np.array(combo_data["num_meas"].tolist())

                            # Sort the data (may not be in order)
                            # gainLoss_ordered = np.flip(np.sort(gainLoss_plot))
                            # gainLoss_ord_meas = gainLoss_meas[np.flip(np.argsort(gainLoss_plot))]
                            gainLoss_ordered = gainLoss_plot[np.argsort(gainLoss_meas)]
                            gainLoss_ord_meas = np.sort(gainLoss_meas)

                            # Compute the number of required measurements for this channel/alpha
                            all_within_loss = (gainLoss_ordered < max_gain_loss)
                            num_within_loss = np.sum(all_within_loss)
                            # if all_models[model_i] == cnn:
                            #     print("{} channel, cnn: {} req. meas".format(all_channel_nlos[chan_plot_i], gainLoss_ord_meas[np.argmax(all_within_loss)]))
                            #     import pdb; pdb.set_trace()
                            
                            if num_within_loss <= 0:
                                continue
                            else:
                                num_required_meas_i[chan_plot_i] = gainLoss_ord_meas[np.argmax(all_within_loss)]
                            

                        # plt.plot(gainLoss_meas, gainLoss_plot, style_plot[type_i][model_i], 
                        #         marker="o", lw=4, label="{}, {}".format(all_beam_combos[model_i], sim_or_exp_labels[type_i]))
                        line_format_i = type_i
                        if (len(sim_or_exp_plot) == 1):
                            line_format_i = 1
                        # Plot the results
                        #import pdb; pdb.set_trace()
                        plt.plot(np.sort(all_channel_alpha), num_required_meas_i[np.argsort(all_channel_alpha)], line_plot[line_format_i][model_i], color=colors_plot[type_i][model_i], 
                                marker="o", lw=4, label="{}, {}".format(all_models_labels[model_i], meas_combo_label))
                plt.title("Number of Required Measurements for {} dB {}th Percentile Gain Loss".format(max_gain_loss, gain_loss_perc))
                plt.xlabel(r"{} Channel Multipath Strength ($\alpha$)".format(sim_or_exp_labels[0]))
                plt.ylabel("Number of Required Meas. (M)")
                plt.grid(True)
                # plt.ylim(np.min(included_num_meas),np.max(included_num_meas))
                if sim_exp == 'sim':
                    if resize_plots:
                        plt.legend(loc="best")
                    else:
                        plt.legend(loc="upper right")
                    # plt.ylim(4, default_num_meas+0.1)
                    plt.ylim(4, 36.1)
                else:
                    if resize_plots:
                        plt.legend(loc="best")
                    else:
                        plt.legend(loc="upper right")
                    # plt.ylim(4, default_num_meas+0.1)
                    plt.ylim(4, 36.1)
                plt.xlim(np.min(all_channel_alpha),np.max(all_channel_alpha))
                if resize_plots:
                    # fig.set_size_inches(6, 2.5)
                    fig.set_size_inches(plot_width, plot_height)
                    fig.subplots_adjust(top=(fig.subplotpars.top - 0.05))
                    fig.subplots_adjust(bottom=0.17)
                plt.draw()
                if SAVE_PLOTS:
                    if resize_plots:
                        plt.savefig(os.path.join(self.results_dir, "{}_small.{}".format(test_title, save_type)))
                    else:
                        plt.savefig(os.path.join(self.results_dir, "{}.{}".format(test_title, save_type)))


            else:
                print("ERROR: INVALID PLOT TYPE \"{}\"".format(plot_type))
        else:
            print("ERROR: INVALID METRIC \"{}\"".format(plot_metric))

        # plt.show()
        if plot_tight_layout:
            plt.tight_layout()




class MLBF_Dataset_Tag:
    #### Class to hold dataset (per file/config) labeling information for indexing and plots
    def __init__(self, test_name, long_tags, short_tags, sim_or_exp_all, 
                 snrs, nlos_rel_pwr, 
                 test_only_flags=[],
                 sim_nums=[], file_tags=[], dates=[], txidxs=[]):
        self.test_name = test_name              # Name used for this test configuration and file name (i.e. short description of the comparison/simulation/experiment)
        self.long_tags = long_tags              # Tag used for long labels and notes
        self.short_tags = short_tags            # Tag used for indexing results (often SNR for simulation and some short label for experiments)
        self.DATA_SNR = snrs                    # List of SNRs used (required for both sims and exps)
        self.DATA_TAGS = file_tags              # Tag used for filenames (often unused for sims)
        self.DATA_DATES = dates                 # List of dates for the test results (often unused for sims)
        self.DATA_TXIDX = txidxs                # List of transmit power indices used (experiments only)
        self.sim_nums = sim_nums                 # Simulation numbers (-1 for experimental data)
        self.NLOS_PATH_REL_PWR = nlos_rel_pwr   # NLOS path relative power compared to the stronges/LOS path
        self.DATA_TEST_ONLY = test_only_flags   # Boolean flags to indicate which files should only be used for testing (not training)

        self.sim_or_exp = []
        for i in np.arange(len(sim_or_exp_all)):
            if (sim_or_exp_all[i] == "sim") or (sim_or_exp_all[i] == "exp"):
                self.sim_or_exp.append(sim_or_exp_all[i])
            else:
                print("ERROR: Invalid selection (#{}) for data type ({}) - must be \"sim\" or \"exp\"".format(i+1, sim_or_exp_all[i]))
        

class Logger(object):
    #### Class to overwrite stdout to log data to file (as well as output to terminal)
    #       - Developed since Windows cmd is annoying -> no "tee" function in cmd 
    def __init__(self, log_file_path, mode="w"):
        # Mode is usually either "w" (overwrite file if it exists) or "a" (append to file if it exists)
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, mode)

    def __del__(self):
        sys.stdout = self.terminal
        self.log_file.close()

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return True


