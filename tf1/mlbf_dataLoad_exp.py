import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from scipy.special import comb
import copy

from mlbf_dataset import MLBF_Dataset, MLBF_Dataset_Tag



class ExpDataConfig:
    #### Simulation Data configuration class - holds all configuration variables

    def __init__(self, dataset_tag, 
                 num_meas, num_beam_combos, nn_arch,
                 use_pn, use_dftsa, use_qpd, 
                 num_meas_per_dft=20,
                 use_linear_scale=True, use_zero_mean=False, use_normalization=True,
                 use_new_default=True, old_default=-100, new_default=-400,
                 exp_data_type='ext',
                 num_dft=65, num_tot_pn=36, num_tot_dftsa=10, num_tot_qpd=10):
        
        ## Commonly reconfigured options
        self.NUM_MEAS = num_meas                        # Number of PN beams/measurements/features to use (5, 10, 15 typically) #TODO: NEED TO CHOOSE THE ANGLES SOMEHOW
        self.NUM_PN_COMBO = num_beam_combos             # Number of random combinations of measurements to run with the algorithm (to ensure the results aren't dependent on beam)(limit NUM_TOTAL_MEAS choose NUM_MEAS)
                                                        # Set the NUM_PN_COMBO to 1 for the first NUM_MEAS PN beams (i.e. NOT random)
        self.NUM_ANGLES_PER_DFT = num_meas_per_dft      # Number of measurements to use for a DFT (per each date; minimum required to be included)
        self.NN_ARCH = nn_arch                          # Options: "CNN", "FCnet" 
                                                        # NN architecture being tested during this round

        
        ## Dataset configurations
        self.DATA_TEST_ONLY = dataset_tag.DATA_TEST_ONLY    # Boolean flags for which datasets to include for testing only (and not training)
        self.EXP_DATA_TYPE = exp_data_type              # Data type used from the experimental results
        

        ## Channel configuration
        self.DATA_DATES = dataset_tag.DATA_DATES        # Dates used for these experiments
        self.DATA_TAGS = dataset_tag.DATA_TAGS          # Tags used for the data files
        self.DATA_TXIDX = dataset_tag.DATA_TXIDX        # Tx power indices used in experiments
        self.DATA_SNR = dataset_tag.DATA_SNR            # Array of SNRs used in experiments (STF-SNR)
        self.NLOS_PATH_REL_PWR = dataset_tag.NLOS_PATH_REL_PWR  # Power level in dB (negative implied)


        ## Sounding configuration
        self.NUM_DFT = num_dft                      # Number of DFT beams measured
        self.NUM_TOTAL_PN_MEAS = num_tot_pn         # Number of total PN beams/measurements/features collected (in a file)
        self.NUM_TOTAL_DFTSA_MEAS = num_tot_dftsa   # Number of total multifinger beams/measurements/features collected (in a file)
        self.NUM_TOTAL_QPD_MEAS = num_tot_qpd       # Number of total QPD/wide beams/measurements/features collected (in a file)

        self.USE_PN_BEAMS = use_pn                  # True if PN beam features should be used
        self.USE_DFTSA_BEAMS = use_dftsa            # True if multifinger (sub-array DFT beams) should be used
        self.USE_QPD_BEAMS = use_qpd                # True if QPD (wide beams) should be used

        self.NUM_DATASETS = len(self.DATA_SNR)      # Number of datasets included in this run
        self.DATA_LINEAR_SCALE = use_linear_scale   # Rescale the data from log to linear (pRx is in dB)
        self.DATA_ZERO_MEAN = use_zero_mean         # Offset the data by the mean features of the training data (same offset used for test data)
        self.DATA_NORMALIZATION = use_normalization # Scale the data (training and testing) to the maximum norm of the training data

        self.DATA_CONVERT_DEFAULT = use_new_default # Change default value in the data (selected by post-processing software)
        self.DATA_DEFAULT_CURRENT = old_default     # Default value assigned by the post-processing software
        self.DATA_DEFAULT_NEW = new_default         # Default value to replace the one assigned by the post-processing software

        #### Constants - Must be changed manually in order to reset values
        # self.DATA_DIR_FILESTEM = "../data/nn_sim{}"
        # self.DATA_DFT_FILESTEM = "{}/DFT_output_nr{}_{}dB_all.csv" #"../data/results_awv0_{}_{}_dft.csv"
        # self.DATA_PN_FILESTEM = "{}/measurement_RSS_nr{}_{}dB_all.csv" #"../data/results_awv0_{}_{}_pn.csv"
        # self.DATA_LABELS_FILESTEM = "{}/DFT_label_nr{}_{}dB_all.csv" #"../data/results_awv0_{}_{}_labels.csv"
        # self.ANGLE_LABEL_FILESTEM = "{}/label_nr{}_{}dB_all.csv"
        # self.DATA_DFTSA_FILESTEM = "{}/measurement_DFTsa_RSS_nr{}_{}dB_all.csv"
        # self.DATA_QPD_FILESTEM = "{}/measurement_QPD_RSS_nr{}_{}dB_all.csv"
        self.DATA_DIR_FILESTEM = "../data"
        self.DATA_DFT_FILESTEM = "{}/{}_{}/results_awv0_{}_dft.csv"
        self.DATA_PN_FILESTEM = "{}/{}_{}/results_awv0_{}_pn.csv"
        self.DATA_QPD_FILESTEM = "{}/{}_{}/results_awv0_{}_qpd.csv"
        self.DATA_DFTSA_FILESTEM = "{}/{}_{}/results_awv0_{}_dftsa.csv"
        self.DATA_LABELS_FILESTEM = "{}/{}_{}/results_awv0_{}_labels.csv"
        self.DATA_STFSNR_DFT_FILESTEM = "{}/{}_{}/results_awv0_{}_stfsnr_dft.csv"
        self.DATA_STFSNR_PN_FILESTEM = "{}/{}_{}/results_awv0_{}_stfsnr_pn.csv"

        self.PN_LABEL_STEM = "{} PN Beams"
        self.DFTSA_LABEL_STEM = "{} SA Beams"
        self.QPD_LABEL_STEM = "{} QPD Beams"
        self.LABEL_DIV = ", "

        self.SAME_DATA = 0                          # Use the same data for all tests (based on test data for MAX_NUM_ANGLES_PER_DFT) and non-randomized training data
        self.MAX_NUM_ANGLES_PER_DFT = 20            # Max number of points per DFT label required (typically 20)



        ## Computed/Prior default values
        self.DATA_DIR = "../data"
        self.SELECT_COMBOS = False                  # True if you want to fix the beam combos (instead of uniformly selected)
        self.PN_USE_NUM = []                        # Lists of the indexes to use for data features
        self.DFTSA_USE_NUM = []
        self.QPD_USE_NUM = []

        # self.TEST_NAME = "exp_{}_{}trainpts_NLOS{}_PN{}_QPD{}_DFTSA{}".format(self.NN_ARCH,             # Name for this test scenario
        #                                                             self.NUM_ANGLES_PER_DFT, 
        #                                                             self.NLOS_PATH_REL_PWR[0], 
        #                                                             1*self.USE_PN_BEAMS, 1*self.USE_QPD_BEAMS, 1*self.USE_DFTSA_BEAMS)
        #                                                             #TODO: FIX THE LABELING FOR THE CHANNEL INFO
        self.TEST_NAME = "{}_exp".format(dataset_tag.test_name)

        print("Data directory: {}".format(self.DATA_DIR))

        self.NUM_DATES = len(self.DATA_DATES)
        self.NUM_TOTAL_MEAS = self.NUM_TOTAL_PN_MEAS + self.NUM_TOTAL_DFTSA_MEAS + self.NUM_TOTAL_QPD_MEAS
        self.NUM_SNRS = len(self.DATA_SNR)
        self.USE_NUM_BEAM_TYPES = self.USE_PN_BEAMS + self.USE_DFTSA_BEAMS + self.USE_QPD_BEAMS
        if (self.USE_NUM_BEAM_TYPES > 2) or (self.USE_NUM_BEAM_TYPES <= 0):
            # Only supports combos of 2 types of beams for now
            print("INVALID COMBINATION OF BEAM TYPES -- RERUN WITH ONLY 1 OR 2")

        self.ALL_USE_BEAMS = {}
        self.ALL_USE_BEAMS["PN"] = -np.ones((self.NUM_PN_COMBO, self.NUM_MEAS))
        self.ALL_USE_BEAMS["QPD"] = -np.ones((self.NUM_PN_COMBO, self.NUM_MEAS))
        self.ALL_USE_BEAMS["DFTSA"] = -np.ones((self.NUM_PN_COMBO, self.NUM_MEAS))
        self.use_rand_combos()


    def specify_combos(self, pn_use_nums, dftsa_use_nums, qpd_use_nums):
        #### Set the data configuration up to specify specific combinations of beams
        self.SELECT_COMBOS = True
        self.PN_USE_NUM = pn_use_nums
        self.DFTSA_USE_NUM = dftsa_use_nums
        self.QPD_USE_NUM = qpd_use_nums
        self.NUM_PN_COMBO = np.max([len(self.PN_USE_NUM), len(self.DFTSA_USE_NUM), len(self.QPD_USE_NUM)])
        # TODO: DATA CHECKING (CHECK LENGTHS BASED ON SELECTED FEATURES)
        # Run the beam reselection function
        self.calculate_combos()


    def use_rand_combos(self):
        #### Set the data configuration to use random combinations of feature beams
        self.SELECT_COMBOS = False
        self.calculate_combos()


    def calculate_combos(self):
        ##### Compute the PN beams to use for each PN combo set (number of combos = NUM_PN_COMBO)
        self.PN_USE_BEAMS = -np.ones((self.NUM_PN_COMBO, self.NUM_MEAS))
        self.DFTSA_USE_BEAMS = -np.ones((self.NUM_PN_COMBO, self.NUM_MEAS))
        self.QPD_USE_BEAMS = -np.ones((self.NUM_PN_COMBO, self.NUM_MEAS))
        self.PN_USE_NUM_BEAMS = np.zeros((self.NUM_PN_COMBO))
        self.DFTSA_USE_NUM_BEAMS = np.zeros((self.NUM_PN_COMBO))
        self.QPD_USE_NUM_BEAMS = np.zeros((self.NUM_PN_COMBO))
        self.codebook_labels = []

        # Compute the indices and numbers of beams to use in each combo
        combo_set = (np.arange(self.NUM_PN_COMBO)*np.floor((self.NUM_MEAS)/(self.NUM_PN_COMBO))+(self.NUM_MEAS%2)+int(self.NUM_MEAS/(2*self.NUM_PN_COMBO))).astype(int)
        if self.USE_NUM_BEAM_TYPES == 1:
            if self.USE_PN_BEAMS:
                # OLD - CHECK VALIDITY
                # Check the number of combos requested
                max_pn_combos = comb(self.NUM_TOTAL_MEAS, self.NUM_MEAS)
                if (self.NUM_PN_COMBO > max_pn_combos):
                    self.NUM_PN_COMBO = max_pn_combos.astype(int)
                # Select the beams to use
                self.PN_USE_NUM_BEAMS = self.NUM_MEAS*np.ones((self.NUM_PN_COMBO)).astype(int)
                if self.NUM_PN_COMBO > 1:
                    # Using more than 1 combo -> randomly choose the combinations of beams
                    for pn_i in np.arange(self.NUM_PN_COMBO):
                        self.PN_USE_BEAMS[pn_i, :] = np.random.choice(self.NUM_TOTAL_PN_MEAS, self.NUM_MEAS, replace=False)
                elif self.NUM_PN_COMBO == 1:
                    # Using only 1 combo -> Use the first NUM_MEAS beams
                    self.PN_USE_BEAMS[0, :] = np.arange(self.NUM_MEAS)
                self.codebook_labels.append(self.PN_LABEL_STEM.format(self.NUM_MEAS))
            elif self.USE_DFTSA_BEAMS:
                # Check the number of combos requested
                max_dftsa_combos = comb(self.NUM_TOTAL_DFTSA_MEAS, self.NUM_MEAS)
                if (self.NUM_PN_COMBO > max_dftsa_combos):
                    self.NUM_PN_COMBO = max_dftsa_combos.astype(int)
                # Select the beams to use
                self.DFTSA_USE_NUM_BEAMS = self.NUM_MEAS*np.ones((self.NUM_PN_COMBO)).astype(int)
                if self.NUM_PN_COMBO > 1:
                    # Using more than 1 combo -> randomly choose the combinations of beams
                    for pn_i in np.arange(self.NUM_PN_COMBO):
                        self.DFTSA_USE_BEAMS[pn_i, :] = np.random.choice(self.NUM_TOTAL_DFTSA_MEAS, self.NUM_MEAS, replace=False)
                elif self.NUM_PN_COMBO == 1:
                    # Using only 1 combo -> Use the first NUM_MEAS beams
                    self.DFTSA_USE_BEAMS[0, :] = (np.arange(self.NUM_MEAS)*np.floor((self.NUM_TOTAL_DFTSA_MEAS)/(self.NUM_MEAS))+int(self.NUM_TOTAL_DFTSA_MEAS/(2*self.NUM_MEAS))).astype(int)
                self.codebook_labels.append(self.DFTSA_LABEL_STEM.format(self.NUM_MEAS))
            else: #self.USE_QPD_BEAMS:
                # Check the number of combos requested
                max_qpd_combos = comb(self.NUM_TOTAL_QPD_MEAS, self.NUM_MEAS)
                if (self.NUM_PN_COMBO > max_qpd_combos):
                    self.NUM_PN_COMBO = max_qpd_combos.astype(int)
                # Select the beams to use
                self.QPD_USE_NUM_BEAMS = self.NUM_MEAS*np.ones((self.NUM_PN_COMBO)).astype(int)
                if self.NUM_PN_COMBO > 1:
                    # Using more than 1 combo -> randomly choose the combinations of beams
                    for pn_i in np.arange(self.NUM_PN_COMBO):
                        self.QPD_USE_BEAMS[pn_i, :] = np.random.choice(self.NUM_TOTAL_QPD_MEAS, self.NUM_MEAS, replace=False)
                elif self.NUM_PN_COMBO == 1:
                    # Using only 1 combo -> Use the first NUM_MEAS beams
                    self.QPD_USE_BEAMS[0, :] = (np.arange(self.NUM_MEAS)*np.floor((self.NUM_TOTAL_QPD_MEAS)/(self.NUM_MEAS))+int(self.NUM_TOTAL_QPD_MEAS/(2*self.NUM_MEAS))).astype(int)
                self.codebook_labels.append(self.QPD_LABEL_STEM.format(self.NUM_MEAS))
                
        elif self.USE_NUM_BEAM_TYPES == 2:
            ### SELECT PN BEAMS + MULTIFINGER BEAMS
            if self.USE_PN_BEAMS and self.USE_DFTSA_BEAMS:
                if not self.SELECT_COMBOS:
                    # Automatically pick a combo of beams
                    self.PN_USE_NUM_BEAMS    = combo_set
                    self.DFTSA_USE_NUM_BEAMS = self.NUM_MEAS - self.PN_USE_NUM_BEAMS
                else:
                    # Use the user selected set of beams
                    self.PN_USE_NUM_BEAMS    = np.array(self.PN_USE_NUM).astype(int)
                    self.DFTSA_USE_NUM_BEAMS = self.NUM_MEAS - self.PN_USE_NUM_BEAMS
                # Assign the beam indices
                for combo_i in np.arange(self.NUM_PN_COMBO):
                    pn_num = self.PN_USE_NUM_BEAMS[combo_i]
                    dftsa_num = self.DFTSA_USE_NUM_BEAMS[combo_i]
                    self.PN_USE_BEAMS[combo_i,np.arange(pn_num)] = np.arange(pn_num)
                    self.DFTSA_USE_BEAMS[combo_i,np.arange(dftsa_num)] = \
                        (np.arange(dftsa_num)*np.floor((self.NUM_TOTAL_DFTSA_MEAS)/(dftsa_num))+int(self.NUM_TOTAL_DFTSA_MEAS/(2*dftsa_num))).astype(int)
                    self.codebook_labels.append(self.PN_LABEL_STEM.format(pn_num)+self.LABEL_DIV+self.DFTSA_LABEL_STEM.format(dftsa_num))
            ### SELECT PN BEAMS + QPD WIDE BEAMS
            elif self.USE_PN_BEAMS and self.USE_QPD_BEAMS:
                if not self.SELECT_COMBOS:
                    # Automatically pick a combo of beams
                    self.PN_USE_NUM_BEAMS    = combo_set
                    self.QPD_USE_NUM_BEAMS   = self.NUM_MEAS - self.PN_USE_NUM_BEAMS
                else:
                    # Use the user selected set of beams
                    self.PN_USE_NUM_BEAMS    = np.array(self.PN_USE_NUM).astype(int)
                    self.QPD_USE_NUM_BEAMS   = self.NUM_MEAS - self.PN_USE_NUM_BEAMS
                # Assign the beam indices
                for combo_i in np.arange(self.NUM_PN_COMBO):
                    qpd_num = self.QPD_USE_NUM_BEAMS[combo_i]
                    pn_num = self.PN_USE_NUM_BEAMS[combo_i]
                    self.QPD_USE_BEAMS[combo_i,np.arange(qpd_num)] = \
                        (np.arange(qpd_num)*np.floor((self.NUM_TOTAL_QPD_MEAS)/(qpd_num))+int(self.NUM_TOTAL_QPD_MEAS/(2*qpd_num))).astype(int)
                    self.PN_USE_BEAMS[combo_i,np.arange(pn_num)] = np.arange(pn_num)
                    self.codebook_labels.append(self.PN_LABEL_STEM.format(pn_num)+self.LABEL_DIV+self.QPD_LABEL_STEM.format(qpd_num))
            ### SELECT MULTIFINGER BEAMS + QPD WIDE BEAMS
            else: #USE_DFTSA_BEAMS and USE_QPD_BEAMS
                if not self.SELECT_COMBOS:
                    # Automatically pick a combo of beams
                    self.QPD_USE_NUM_BEAMS    = combo_set
                    if self.NUM_MEAS == 4:
                        self.QPD_USE_NUM_BEAMS = self.QPD_USE_NUM_BEAMS + 1  # Heuristic to deal with edge case
                    self.DFTSA_USE_NUM_BEAMS  = self.NUM_MEAS - self.QPD_USE_NUM_BEAMS
                else:
                    # Use the user selected set of beams
                    self.QPD_USE_NUM_BEAMS    = np.array(self.QPD_USE_NUM).astype(int)
                    self.DFTSA_USE_NUM_BEAMS  = self.NUM_MEAS - self.QPD_USE_NUM_BEAMS
                # Assign the beam indices
                for combo_i in np.arange(self.NUM_PN_COMBO):
                    qpd_num = self.QPD_USE_NUM_BEAMS[combo_i]
                    dftsa_num = self.DFTSA_USE_NUM_BEAMS[combo_i]
                    self.QPD_USE_BEAMS[combo_i,np.arange(qpd_num)] = \
                        (np.arange(qpd_num)*np.floor((self.NUM_TOTAL_QPD_MEAS)/(qpd_num))+int(self.NUM_TOTAL_QPD_MEAS/(2*qpd_num))).astype(int)
                    self.DFTSA_USE_BEAMS[combo_i,np.arange(dftsa_num)] = \
                        (np.arange(dftsa_num)*np.floor((self.NUM_TOTAL_DFTSA_MEAS)/(dftsa_num))+int(self.NUM_TOTAL_DFTSA_MEAS/(2*dftsa_num))).astype(int)
                    self.codebook_labels.append(self.QPD_LABEL_STEM.format(qpd_num)+self.LABEL_DIV+self.DFTSA_LABEL_STEM.format(dftsa_num))
                    
        else:
            # Invalid number of combos -> error out
            print("INVALID NUMBER OF PN MEASUREMENTS {} -- RERUN WITH A NEW NUMBER".format(self.NUM_PN_COMBO))
            
        self.PN_USE_BEAMS = self.PN_USE_BEAMS.astype(int)
        self.DFTSA_USE_BEAMS = self.DFTSA_USE_BEAMS.astype(int)
        self.QPD_USE_BEAMS = self.QPD_USE_BEAMS.astype(int)

        # Save and print codebook info
        self.ALL_USE_BEAMS["PN"] = self.PN_USE_BEAMS
        self.ALL_USE_BEAMS["QPD"] = self.QPD_USE_BEAMS
        self.ALL_USE_BEAMS["DFTSA"] = self.DFTSA_USE_BEAMS

        print(self.codebook_labels)     # Labels for each set of codebook entry combos
        print("PN beam indices")
        print(self.PN_USE_BEAMS)
        print("QPD beam indices")
        print(self.QPD_USE_BEAMS)
        print("Multifinger beam indices")
        print(self.DFTSA_USE_BEAMS)
    
    
    def select_data(self, plot_meas_nums=True, print_summaries=True):
        #### Function to select the DFT beam indices to use (i.e. check if they have enough data)
        print("Checking for sufficient data; Selecting valid labels...")

        ## Loop through each test date and determine the data to use
        dft_use_all = np.zeros((self.NUM_SNRS, self.NUM_DFT))
        dft_use_all_num = np.zeros((self.NUM_SNRS, self.NUM_DFT))

        for snr_ind in np.arange(self.NUM_SNRS):

            ## Get the file names
            SNR_i = self.DATA_SNR[snr_ind]
            print("Processing date: {}, TXIDX: {}, SNR: {}, Type: {}".format(self.DATA_DATES[snr_ind], self.DATA_TXIDX[snr_ind], SNR_i, self.EXP_DATA_TYPE))
            dft_file = self.DATA_DFT_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            pn_file = self.DATA_PN_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            labels_file = self.DATA_LABELS_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            dftsa_file = self.DATA_DFTSA_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            qpd_file = self.DATA_QPD_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)

            ## Extract the data into Pandas dataframes
            # Number of rows = number of measurements/physical angles
            # Number of columns = number of beams (for DFT and PN; labels should only have 1)
            df_dft = pd.read_csv(dft_file, header=None)
            df_pn = pd.read_csv(pn_file, header=None)
            df_labels = pd.read_csv(labels_file, header=None)
            df_dftsa = pd.read_csv(dftsa_file, header=None)
            df_qpd = pd.read_csv(qpd_file, header=None)
            
            if print_summaries:
                print("\tDFT data: ", df_dft.shape)
                print("\tPN data:  ", df_pn.shape)
                print("\tLabels:   ", df_labels.shape)
                print("\tMultifinger data: ", df_dftsa.shape)
                print("\tQPD data:         ", df_qpd.shape)

            ## Pickout the data to use
            u, c = np.unique(df_labels, return_counts=True)
            if plot_meas_nums:
                plt.figure()
                plt.plot(u, c)
                plt.title("Number of Measurements for Each DFT Beam, Date: {}".format(self.DATA_DATES[snr_ind]))
                plt.xlabel("DFT Beam Index")
                plt.ylabel("Number of Labeled Angles")
                plt.grid(True)
                plt.xlim(0, 63)
                plt.axhline(y=self.NUM_ANGLES_PER_DFT, color='r', linestyle='-')

            if not self.DATA_TEST_ONLY[snr_ind]:
                # Pick the DFT indices just based on meeting the minimum number of angles required
                dft_valid_mask = u[c >= self.NUM_ANGLES_PER_DFT].astype(int)
                dft_use_all[snr_ind, dft_valid_mask] = True
                dft_use_all_num[snr_ind, dft_valid_mask] = c[c >= self.NUM_ANGLES_PER_DFT]
            else:
                # For test-only data, say any DFT beam is OK - TODO: IMPROVE THIS CODE
                dft_valid_mask = u[c >= 0].astype(int)
                dft_use_all[snr_ind, :] = True
                dft_use_all_num[snr_ind, dft_valid_mask] = c[c >= 0]
            
        pick_dft = np.all(dft_use_all, 0)
        self.dft_use = np.nonzero(pick_dft)[0]
        self.dft_use_num = dft_use_all_num[:,pick_dft].astype(int)
        self.NUM_CLASSES = len(self.dft_use)
        if print_summaries:
            print("Num DFT beams to use: ", self.NUM_CLASSES)
            print("Num samples to use:   ", np.sum(self.dft_use_num))
            print("Num PN beam combos:   ", self.NUM_PN_COMBO)
            print("\tNum PN beams used:          ", self.PN_USE_NUM_BEAMS)
            print("\tNum Multifinger beams used: ", self.DFTSA_USE_NUM_BEAMS)
            print("\tNum QPD wide beams used:    ", self.QPD_USE_NUM_BEAMS)
            if self.NUM_PN_COMBO == 1:
                print("\tNOT USING RANDOM BEAMS -> FIRST {} PN MEASUREMENTS USED!".format(self.NUM_MEAS))
        print("Test Name: {}".format(self.TEST_NAME))


    def create_datasets(self, plot_meas_nums=True, print_summaries=True):
        #### Create the dataset arrays given the settings provided
        #       - plot_meas_nums  - Plot the data measurement distribution by label (Only used for the data selection)
        #       - print_summaries - Print information about the dataset (Used for both data selection and the actual dataset creation)

        ## Filter the data for valid labels
        self.select_data(plot_meas_nums=plot_meas_nums, print_summaries=print_summaries)

        ## Run the actual dataset structure creation
        print("Creating dataset...")

        # Running dictonaries for the data
        train_data_dict = {}
        test_data_dict = {}
        #val_data_dict = {}
        train_labels_dict = {}
        test_labels_dict = {}
        #val_labels_dict = {}
        train_classes_dict = {}
        test_classes_dict = {}
        #val_classes_dict = {}
        dft_rssi_dict = {}

        # The actual loop
        for snr_ind in np.arange(self.NUM_SNRS):
        #for date_i in np.arange(NUM_DATES):

            ## Get the file names
            SNR_i = self.DATA_SNR[snr_ind]
            print("Processing date: {}, TXIDX: {}, SNR: {}, Type: {}".format(self.DATA_DATES[snr_ind], self.DATA_TXIDX[snr_ind], SNR_i, self.EXP_DATA_TYPE))
            dft_file = self.DATA_DFT_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            pn_file = self.DATA_PN_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            labels_file = self.DATA_LABELS_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            dftsa_file = self.DATA_DFTSA_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            qpd_file = self.DATA_QPD_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            # stfsnr_dft_file = self.DATA_STFSNR_DFT_FILESTEM.format(self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            # stfsnr_pn_file = self.DATA_STFSNR_PN_FILESTEM.format(self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)

            ## Extract the data into Pandas dataframes
            # Number of rows = number of measurements/physical angles
            # Number of columns = number of beams (for DFT and PN; labels should only have 1)
            df_dft = pd.read_csv(dft_file, header=None)
            df_pn = pd.read_csv(pn_file, header=None)
            df_labels = pd.read_csv(labels_file, header=None)
            df_dftsa = pd.read_csv(dftsa_file, header=None)
            df_qpd = pd.read_csv(qpd_file, header=None)
            # df_stfsnr_dft = pd.read_csv(stfsnr_dft_file, header=None)
            # df_stfsnr_pn = pd.read_csv(stfsnr_pn_file, header=None)
            
            print("\tDFT data: ", df_dft.shape)
            print("\tPN data:  ", df_pn.shape)
            print("\tLabels:   ", df_labels.shape)
            print("\tMultifinger data: ", df_dftsa.shape)
            print("\tQPD data:         ", df_qpd.shape)
            arr_pn = df_pn.to_numpy()
            arr_dftsa = df_dftsa.to_numpy()
            arr_qpd = df_qpd.to_numpy()
            arr_dft = df_dft.to_numpy()
            # arr_stfsnr_dft = df_stfsnr_dft.to_numpy()
            # arr_stfsnr_pn = df_stfsnr_pn.to_numpy()
            
        #     ## Pickout the data to use
        #     u, c = np.unique(df_labels, return_counts=True)
        #     plt.figure()
        #     plt.plot(u, c)
        #     plt.title("Number of Measurements for Each DFT Beam, Date: {}".format(SNR_i))
        #     plt.xlabel("DFT Beam Index")
        #     plt.ylabel("Number of Labeled Angles")
        #     plt.grid(True)
        #     plt.xlim(0, 63)
        #     plt.axhline(y=NUM_ANGLES_PER_DFT, color='r', linestyle='-')

        #     # Pick the DFT indices just based on meeting the minimum number of angles required
        #     dft_use = u[c >= NUM_ANGLES_PER_DFT].astype(int)
        #     dft_use_num = c[c >= NUM_ANGLES_PER_DFT]
            
            # Running dictionary for the data for each PN beam combo
            train_data_datei_dict = {}
            test_data_datei_dict = {}
            #val_data_datei_dict = {}
            train_labels_datei_dict = {}
            test_labels_datei_dict = {}
            #val_labels_datei_dict = {}
            train_classes_datei_dict = {}
            test_classes_datei_dict = {}
            dft_rssi_datei_dict = {}
            
            # Loop through all the PN beam combos
            for pn_i in np.arange(self.NUM_PN_COMBO):

                # Choose the points randomly from the points available (per each used DFT beam)
                train_data = np.array([])
                train_labels = np.array([])
                test_data = np.array([])
                test_labels = np.array([])
                train_classes = np.array([])
                test_classes = np.array([])
                test_dftrssi = np.array([])
                
                for dft_i in np.arange(len(self.dft_use)):

                    # Find all potential PN beam measurements to use for this DFT beam label
                    wh = np.where(df_labels == self.dft_use[dft_i])
                    wh_num_pts = wh[0].shape[0]
                    
                    # Choose the PN beams
                    pn_num = self.PN_USE_NUM_BEAMS[pn_i]
                    if pn_num > 0:
                        # Sort out the measurements desired
                        pn_all_i_temp = arr_pn[wh[0], :]
                        # Sort out the beams desired
                        pn_all_i = pn_all_i_temp[:,self.PN_USE_BEAMS[pn_i,np.arange(pn_num)]]
                    else:
                        # No beams used; add a filler matrix
                        pn_all_i = np.zeros((wh_num_pts,0))
                        
                    # Choose the multifinger beams
                    dftsa_num = self.DFTSA_USE_NUM_BEAMS[pn_i]            
                    if dftsa_num > 0:
                        # Sort out the measurements desired
                        dftsa_all_i_temp = arr_dftsa[wh[0], :]
                        # Sort out the beams desired
                        dftsa_all_i = dftsa_all_i_temp[:,self.DFTSA_USE_BEAMS[pn_i,np.arange(dftsa_num)]]
                    else:
                        # No beams used; add a filler matrix
                        dftsa_all_i = np.zeros((wh_num_pts,0))
                        
                    # Choose the QPD wide beams
                    qpd_num = self.QPD_USE_NUM_BEAMS[pn_i]
                    if qpd_num > 0:
                        # Sort out the measurements desired
                        qpd_all_i_temp = arr_qpd[wh[0], :]
                        # Sort out the beams desired
                        qpd_all_i = qpd_all_i_temp[:,self.QPD_USE_BEAMS[pn_i,np.arange(qpd_num)]]
                    else:
                        # No beams used; add a filler matrix
                        qpd_all_i = np.zeros((wh_num_pts,0))
                        
                    # Combine the feature beams (PN, multifinger, and QPD) into a single codebook result
                    features_all_i = np.concatenate((pn_all_i, dftsa_all_i, qpd_all_i), axis=1)
                    
                    # Choose the DFT beams
                    dft_all_i = arr_dft[wh[0], :]
                    #print("------ PN select size: {}".format(pn_all_i.shape))
        #             print(dft_use[dft_i])
        #             print(len(wh[0]))
        #             print(pn_all_i.shape)

                    # Randomly pick the indices to include as training data; rest is test data
                    pn_all_ind = np.arange(self.dft_use_num[snr_ind, dft_i])                             # List of indices for all data points with this DFT label
                    if not self.SAME_DATA:
                        if not self.DATA_TEST_ONLY[snr_ind]:
                            # Select the random choices for the normal case (data used for both training and testing)
                            pn_choice_ind = np.random.choice(pn_all_ind, self.NUM_ANGLES_PER_DFT, replace=False) # Pick out the training data indices
                            pn_nchoice_ind = np.delete(pn_all_ind, pn_choice_ind)                           # Indices for the data points used for testing
                        else:
                            # Relegate all data to testing only
                            pn_choice_ind = np.array([])                          # Choose no data for training
                            pn_nchoice_ind = pn_all_ind                           # All data goes to testing
                    else:
                        pn_choice_max_ind = np.arange(self.MAX_NUM_ANGLES_PER_DFT)
                        pn_choice_ind = np.arange(self.NUM_ANGLES_PER_DFT)
                        pn_nchoice_ind = np.delete(pn_all_ind, pn_choice_max_ind) # Indices for the data points used for testing
                    
                    if (pn_choice_ind.size > 0):
                        pn_choice = features_all_i[pn_choice_ind, :]              # Features (PN, multi, QPD) for the points used for training
                    else:
                        pn_choice = np.array([])


                    #pn_choice_ind = np.random.choice(pn_all_ind, self.NUM_ANGLES_PER_DFT, replace=False) # Pick out the training data indices
                    #pn_choice = features_all_i[pn_choice_ind, :]                                    # Features (PN, multi, QPD) for the points used for training
                    #pn_nchoice_ind = np.delete(pn_all_ind, pn_choice_ind)                           # Indices for the data points used for testing
                    
                    pn_nchoice = features_all_i[pn_nchoice_ind, :]                                  # Features (PN, multi, QPD) for the points used for testing
                    dft_nchoice = dft_all_i[pn_nchoice_ind, :]
                    choice_labels = self.dft_use[dft_i] * np.ones((self.NUM_ANGLES_PER_DFT, 1))
                    #nchoice_labels = self.dft_use[dft_i] * np.ones((self.dft_use_num[snr_ind, dft_i] - self.NUM_ANGLES_PER_DFT, 1))
                    choice_classes = dft_i * np.ones((self.NUM_ANGLES_PER_DFT, 1))
                    #nchoice_classes = dft_i * np.ones((self.dft_use_num[snr_ind, dft_i] - self.NUM_ANGLES_PER_DFT, 1))

                    if not self.SAME_DATA:
                        if not self.DATA_TEST_ONLY[snr_ind]:
                            # Pick out the test data labels and classes required
                            nchoice_labels = self.dft_use[dft_i] * np.ones((self.dft_use_num[snr_ind, dft_i] - self.NUM_ANGLES_PER_DFT, 1))
                            nchoice_classes = dft_i * np.ones((self.dft_use_num[snr_ind, dft_i] - self.NUM_ANGLES_PER_DFT, 1))
                        else:
                            # If data only used for test data, setup a full length array of labels
                            nchoice_labels = self.dft_use[dft_i] * np.ones((self.dft_use_num[snr_ind, dft_i], 1))
                            nchoice_classes = dft_i * np.ones((self.dft_use_num[snr_ind, dft_i], 1))
                    else:
                        nchoice_labels = self.dft_use[dft_i] * np.ones((self.dft_use_num[snr_ind, dft_i] - self.MAX_NUM_ANGLES_PER_DFT, 1))
                        nchoice_classes = dft_i * np.ones((self.dft_use_num[snr_ind, dft_i] - self.MAX_NUM_ANGLES_PER_DFT, 1))


                    if dft_i == 0:
                        train_data = pn_choice
                        train_labels = choice_labels
                        test_data = pn_nchoice
                        test_labels = nchoice_labels
                        train_classes = choice_classes
                        test_classes = nchoice_classes
                        test_dftrssi = dft_nchoice
                    else:
                        train_data = np.vstack((train_data, pn_choice))
                        train_labels = np.vstack((train_labels, choice_labels))
                        test_data = np.vstack((test_data, pn_nchoice))
                        test_labels = np.vstack((test_labels, nchoice_labels))
                        train_classes = np.vstack((train_classes, choice_classes))
                        test_classes = np.vstack((test_classes, nchoice_classes))
                        test_dftrssi = np.vstack((test_dftrssi, dft_nchoice))

                # Deal with invalid training sets used for test-only data
                if self.DATA_TEST_ONLY[snr_ind]:
                    train_data = np.empty((0,self.NUM_MEAS))
                    train_labels = np.empty((0,1))
                    train_classes = np.empty((0,1))
                
                # Store the results in the date dictionary
                print("\t\tTraining data: {}; labels: {}; classes: {} - PN combo {}".format(train_data.shape, train_labels.shape, train_classes.shape, pn_i))
                print("\t\tTesting data:  {}; labels: {}; classes: {} - PN combo {}".format(test_data.shape, test_labels.shape, test_classes.shape, pn_i))
                train_data_datei_dict[pn_i] = train_data
                test_data_datei_dict[pn_i] = test_data
                #val_data_datei_dict[pn_i] = val_data
                train_labels_datei_dict[pn_i] = train_labels.astype(int)
                test_labels_datei_dict[pn_i] = test_labels.astype(int)
                #val_labels_datei_dict[pn_i] = val_labels
                train_classes_datei_dict[pn_i] = train_classes.astype(int)
                test_classes_datei_dict[pn_i] = test_classes.astype(int)
                
                # Save the DFT beam RSSI for future gain loss performance evaluation
                dft_rssi_datei_dict[pn_i] = test_dftrssi
                
            # Store the results in the overall dictionaries
            print("\tTraining dictionaries - data: {}; labels: {}; classes: {}".format(len(train_data_datei_dict), len(train_labels_datei_dict), len(train_classes_datei_dict)))
            print("\tTesting dictionaries - data:  {}; labels: {}; classes: {}".format(len(test_data_datei_dict), len(test_labels_datei_dict), len(test_classes_datei_dict)))
            train_data_dict[snr_ind] = train_data_datei_dict
            test_data_dict[snr_ind] = test_data_datei_dict
            #val_data_dict[snr_ind] = val_data_datei_dict
            train_labels_dict[snr_ind] = train_labels_datei_dict
            test_labels_dict[snr_ind] = test_labels_datei_dict
            #val_labels_dict[snr_ind] = val_labels_datei_dict
            train_classes_dict[snr_ind] = train_classes_datei_dict
            test_classes_dict[snr_ind] = test_classes_datei_dict
            dft_rssi_dict[snr_ind] = dft_rssi_datei_dict
            
        ## Final variable cleanup
        train_data_all = {}
        train_labels_all = {}
        train_classes_all = {}
        # val_data_all = {}
        # val_labels_all = {}
        # val_classes_all = {}
        test_data_all = {}
        test_labels_all = {}
        test_classes_all = {}
        test_dftrssi_all = {}

        print("\nCreating total training/validation/test datasets:")
        for pn_i in np.arange(self.NUM_PN_COMBO):
            # Running variables
            train_data_pni = np.array([])
            train_labels_pni = np.array([])
            train_classes_pni = np.array([])
            # val_data_pni = np.array([])
            # val_labels_pni = np.array([])
            # val_classes_pni = np.array([])
            test_data_pni = np.array([])
            test_labels_pni = np.array([])
            test_classes_pni = np.array([])
            test_dftrssi_pni = np.array([])
            importeddata = False
            
            # Combine the data for each set of PN beams/measurements
            for snr_i in np.arange(len(train_data_dict)):
                #print("\ttrain: {}, val: {}".format(train_data_dict[snr_i][pn_i].shape, val_data_dict[snr_i][pn_i].shape))
                print("\ttrain: {}".format(train_data_dict[snr_i][pn_i].shape))
                if not importeddata:
                    train_data_pni = train_data_dict[snr_i][pn_i]
                    train_labels_pni = train_labels_dict[snr_i][pn_i]
                    train_classes_pni = train_classes_dict[snr_i][pn_i]
            #         val_data_pni = val_data_dict[snr_i][pn_i]
            #         val_labels_pni = val_labels_dict[snr_i][pn_i]
                    test_data_pni = test_data_dict[snr_i][pn_i]
                    test_labels_pni = test_labels_dict[snr_i][pn_i]
                    test_classes_pni = test_classes_dict[snr_i][pn_i]
                    test_dftrssi_pni = dft_rssi_dict[snr_i][pn_i]
                    importeddata = True
                else:
                    train_data_pni = np.vstack((train_data_pni, train_data_dict[snr_i][pn_i]))
                    train_labels_pni = np.vstack((train_labels_pni, train_labels_dict[snr_i][pn_i]))
                    train_classes_pni = np.vstack((train_classes_pni, train_classes_dict[snr_i][pn_i]))
            #         val_data_pni = np.vstack((val_data_pni, val_data_dict[snr_i][pn_i]))
            #         val_labels_pni = np.vstack((val_labels_pni, val_labels_dict[snr_i][pn_i]))
                    test_data_pni = np.vstack((test_data_pni, test_data_dict[snr_i][pn_i]))
                    test_labels_pni = np.vstack((test_labels_pni, test_labels_dict[snr_i][pn_i]))
                    test_classes_pni = np.vstack((test_classes_pni, test_classes_dict[snr_i][pn_i]))
                    test_dftrssi_pni = np.vstack((test_dftrssi_pni, dft_rssi_dict[snr_i][pn_i]))
            
            # Store the combined data for this set of features into the overall dictionary
            train_data_all[pn_i] = train_data_pni
            train_labels_all[pn_i] = train_labels_pni
            train_classes_all[pn_i] = train_classes_pni
            test_data_all[pn_i] = test_data_pni
            test_labels_all[pn_i] = test_labels_pni
            test_classes_all[pn_i] = test_classes_pni
            test_dftrssi_all[pn_i] = test_dftrssi_pni

        ## Data scaling and normalization
        print("\nRunning data scaling and normalization...")
        self.NUM_TRAIN = np.zeros((self.NUM_PN_COMBO, 1))
        for pn_i in np.arange(self.NUM_PN_COMBO):
            print("\tTraining data range: ({}, {}) --- PN set {}".format(np.min(train_data_all[pn_i]), np.max(train_data_all[pn_i]), pn_i))
            if self.DATA_CONVERT_DEFAULT:
                print("\t\tChanging default measurement value - Old: {}; New: {}...".format(self.DATA_DEFAULT_CURRENT, self.DATA_DEFAULT_NEW))
                num_changed = np.sum(train_data_all[pn_i] == self.DATA_DEFAULT_CURRENT)
                train_data_all[pn_i][train_data_all[pn_i] == self.DATA_DEFAULT_CURRENT] = self.DATA_DEFAULT_NEW
                test_data_all[pn_i][test_data_all[pn_i] == self.DATA_DEFAULT_CURRENT] = self.DATA_DEFAULT_NEW
                print("\t\tTotal of {} measurements changed".format(num_changed))
            
            if self.DATA_LINEAR_SCALE:
                print("\t\tUsing linear scale data...")
                # train_data_all[pn_i] = 10**(train_data_all[pn_i]/10)
                # test_data_all[pn_i] = 10**(test_data_all[pn_i]/10)
                # for snr_i in np.arange(len(test_data_dict)):
                #     test_data_dict[snr_i][pn_i] = 10**(test_data_dict[snr_i][pn_i]/10)
                train_data_all[pn_i] = 10**(train_data_all[pn_i]/20)
                test_data_all[pn_i] = 10**(test_data_all[pn_i]/20)
                for snr_i in np.arange(len(test_data_dict)):
                    test_data_dict[snr_i][pn_i] = 10**(test_data_dict[snr_i][pn_i]/20)

            if self.DATA_ZERO_MEAN:
                train_mean = np.mean(train_data_all[pn_i], 0)
                print("\t\tUsing zero-mean offset; offset by {}...".format(train_mean))
                train_data_all[pn_i] = train_data_all[pn_i] - train_mean
                test_data_all[pn_i] = test_data_all[pn_i] - train_mean
                for snr_i in np.arange(len(test_data_dict)):
                    test_data_dict[snr_i][pn_i] = test_data_dict[snr_i][pn_i] - train_mean

            if self.DATA_NORMALIZATION:
                data_scale = np.max(np.linalg.norm(train_data_all[pn_i], 2, 1))
                print("\t\tData scale factor (max norm): {}...".format(data_scale))
                train_data_all[pn_i] = train_data_all[pn_i]/data_scale
                test_data_all[pn_i] = test_data_all[pn_i]/data_scale
                for snr_i in np.arange(len(test_data_dict)):
                    test_data_dict[snr_i][pn_i] = test_data_dict[snr_i][pn_i]/data_scale
                print("\t\tNew training data range: {}, {}".format(np.min(train_data_all[pn_i]), np.max(train_data_all[pn_i])))

            self.NUM_TRAIN[pn_i] = int(train_data_all[pn_i].shape[0])
            print("\t\tTotal training dataset size (PN set {}): {}".format(pn_i, train_data_all[pn_i].shape))
            print("\t\tTotal test dataset size     (PN set {}): {}".format(pn_i, test_data_all[pn_i].shape))
            # NUM_VAL = int(val_data_all.shape[0])
            # print("Total validation dataset size: {}".format(val_data_all.shape))
            
        # Save the total data to a new dictionary key (useful for test loops)
        train_data_dict['ALL'] = train_data_all
        train_labels_dict['ALL'] = train_labels_all
        train_classes_dict['ALL'] = train_classes_all
        test_data_dict['ALL'] = test_data_all
        test_labels_dict['ALL'] = test_labels_all
        test_classes_dict['ALL'] = test_classes_all
        dft_rssi_dict['ALL'] = test_dftrssi_all

        # Save the dataset to the MLBF_Dataset class and return a MLBF_Dataset object
        #   Note: The dataset is NOT included in the SimDataConfig class
        #         Instead, the opposite is true (this SimDataConfig is refered to in)
        dataset = MLBF_Dataset('sim', copy.deepcopy(self))
        dataset.store_train_data(train_data_dict, train_labels_dict, train_classes_dict)
        dataset.store_test_data(test_data_dict, test_labels_dict, test_classes_dict, dft_rssi_dict)
        return dataset


    def snr_statistics(self, num_nlos_paths, min_beam_idx_sep, print_summaries):
        ## Compute statistics on the DFT label beam and PN beam STF-SNR
        STFSNR_PERCENTILES = np.arange(0, 110, 10)
        REL_PATH_GAIN_PERCENTILES = np.arange(0, 110, 10)
        NUM_NLOS = num_nlos_paths                       # Number of NLOS paths to consider (does NOT count the LOS path)
        MIN_PATH_SEPARATION = min_beam_idx_sep          # Minimum number of DFT beam indices of separation required between paths
                                                        #   Typically 64 beams for 45 deg/36 elements -> 0.71 deg separation
                                                        #   Oversampled in spatial domain; should use at least 2-3 index separation
        DEFAULT_MIN_VAL = self.DATA_DEFAULT_CURRENT     # Default value to use to void out used beams (should be <= the smallest expected value)
        print_rel_path_gain_percentiles = print_summaries
        stfsnr_dft_dict = {}                            # Store the STF-SNR for the label DFT beams by beam label
        stfsnr_pn_dict = {}                             # Store the STF-SNR for the selected PN beams by beam label

        for snr_ind in np.arange(self.NUM_SNRS):

            ## Get the file names
            print("Processing date: {}, TXIDX: {}, SNR: {}, Type: {}".format(self.DATA_DATES[snr_ind], self.DATA_TXIDX[snr_ind], self.DATA_SNR[snr_ind], self.EXP_DATA_TYPE))
            dft_file = self.DATA_DFT_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            # pn_file = self.DATA_PN_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            # qpd_file = self.DATA_QPD_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            # dftsa_file = self.DATA_DFTSA_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            labels_file = self.DATA_LABELS_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            stfsnr_dft_file = self.DATA_STFSNR_DFT_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            stfsnr_pn_file = self.DATA_STFSNR_PN_FILESTEM.format(self.DATA_DIR_FILESTEM, self.DATA_DATES[snr_ind], self.DATA_TAGS[snr_ind], self.EXP_DATA_TYPE)
            
            # SNR_i = self.DATA_SNR[snr_ind]

            ## Extract the data into Pandas dataframes
            # Number of rows = number of measurements/physical angles
            # Number of columns = number of beams (for DFT and PN; labels should only have 1)
            df_dft = pd.read_csv(dft_file, header=None)
            df_labels = pd.read_csv(labels_file, header=None)
            df_stfsnr_dft = pd.read_csv(stfsnr_dft_file, header=None)
            df_stfsnr_pn = pd.read_csv(stfsnr_pn_file, header=None)
            
            arr_dft = df_dft.to_numpy()
            arr_stfsnr_dft = df_stfsnr_dft.to_numpy()
            arr_stfsnr_pn = df_stfsnr_pn.to_numpy()
            
            # Running dictionaries for the data for each PN beam combo
            stfsnr_dft_arr_i = {}
            stfsnr_pn_arr_i = {}
            stfsnr_dft_date_i = np.array([])
            sftsnr_pn_date_i = np.array([])
            prx_dft_date_i = np.array([])
            prx_dft_all_date_i = np.array([])
            
            # Loop through all the DFT beam labels (to determine which points are actually used)
            pn_combo_i = 0
            for dft_i in np.arange(len(self.dft_use)):

                # Find all potential PN beam measurements to use for this DFT beam label
                wh = np.where(df_labels == self.dft_use[dft_i])
                dft_all_i = arr_dft[wh[0], :]
                pn_ss_i_temp = arr_stfsnr_pn[wh[0], :]
                stfsnr_pn_arr_i[self.dft_use[dft_i]] = pn_ss_i_temp[:,self.PN_USE_BEAMS[pn_combo_i]]
                stfsnr_dft_arr_i[self.dft_use[dft_i]] = arr_stfsnr_dft[wh[0], self.dft_use[dft_i]]
                
                #print(stfsnr_dft_arr_i[dft_use[dft_i]].shape)
                # Add to the overall lists of data
                if dft_i == 0:
                    stfsnr_pn_date_i = stfsnr_pn_arr_i[self.dft_use[dft_i]]
                    stfsnr_dft_date_i = stfsnr_dft_arr_i[self.dft_use[dft_i]]
                    prx_dft_date_i = dft_all_i[:, self.dft_use[dft_i]]
                    prx_dft_all_date_i = dft_all_i
                else:
                    stfsnr_pn_date_i = np.vstack((stfsnr_pn_date_i, stfsnr_pn_arr_i[self.dft_use[dft_i]]))
                    stfsnr_dft_date_i = np.concatenate((stfsnr_dft_date_i, stfsnr_dft_arr_i[self.dft_use[dft_i]]))
                    prx_dft_date_i = np.concatenate((prx_dft_date_i, dft_all_i[:, self.dft_use[dft_i]]))
                    prx_dft_all_date_i = np.concatenate((prx_dft_all_date_i, dft_all_i))
                
                #print("dft {} -> {}".format(stfsnr_dft_arr_i[self.dft_use[dft_i]].shape, stfsnr_dft_date_i.shape))
                #print("pn  {} -> {}".format(stfsnr_pn_arr_i[self.dft_use[dft_i]].shape, stfsnr_pn_date_i.shape))
                
            stfsnr_dft_dict[snr_ind] = stfsnr_dft_arr_i
            stfsnr_pn_dict[snr_ind] = stfsnr_pn_arr_i
            print(stfsnr_dft_date_i.shape)
            print(stfsnr_pn_date_i.shape)
            
            # Compute the statistics for this day's data
            stfsnr_dft_perc_i = np.percentile(stfsnr_dft_date_i, STFSNR_PERCENTILES, interpolation='lower')
            print("\tSTF-SNR (DFT labels): Test {}".format(self.DATA_TAGS[snr_ind]))
            print("\t\tMean: {}; Std-dev: {}".format(np.mean(stfsnr_dft_date_i), np.std(stfsnr_dft_date_i)))
            print("\t\tPercentiles:")
            print(np.vstack((STFSNR_PERCENTILES, stfsnr_dft_perc_i)).T)

            stfsnr_pn_perc_i = np.percentile(stfsnr_pn_date_i, STFSNR_PERCENTILES, interpolation='lower')
            print("\tSTF-SNR (PN beams): Test {}".format(self.DATA_TAGS[snr_ind]))
            print("\t\tMean: {}; Std-dev: {}".format(np.mean(stfsnr_pn_date_i), np.std(stfsnr_pn_date_i)))
            print("\t\tPercentiles:")
            print(np.vstack((STFSNR_PERCENTILES, stfsnr_pn_perc_i)).T)
            
            # Compute the statistics of the 2nd best DFT beams
            if print_rel_path_gain_percentiles and (NUM_NLOS > 0):
                # Sort the PRX values and pick which columns to look at 
                #   Need to find reasonable separation since the DFT beams are oversampled
                
                # Array of pRx values
                path_all_prx = np.zeros((prx_dft_all_date_i.shape[0],NUM_NLOS+1))
                path_all_idx = -np.ones((prx_dft_all_date_i.shape[0],NUM_NLOS+1))
                path_rel_prx = np.zeros((prx_dft_all_date_i.shape[0],NUM_NLOS))

                # Loop to determine the NLOS paths
                prx_remain = prx_dft_all_date_i         # Running variable for choosing different NLOS beams
                prev_best_prx = np.array([])            # Running variable for the current pRx array for the previous best beams
                overall_best_prx = np.array([])         # Storage for the best pRx found
                for path_i in np.arange(NUM_NLOS+1):
                    # Sort the DFT beams by pRx
                    a_s = np.sort(prx_remain, axis=1)
                    # Find the indices of the sorted beams
                    a_s_idx = np.argsort(prx_remain, axis=1)
                    # Best beam indices
                    a_best_idx = a_s_idx[:,-1]
                    # Best beam pRx
                    a_best_prx = np.array([prx_remain[point_i,a_best_idx[point_i]] for point_i in np.arange(prx_dft_all_date_i.shape[0])])

                    # Store the difference in path pRx
                    if prev_best_prx.size != 0:
                        # Compute the difference in path pRx
                        path_rel_prx[:,path_i-1] = overall_best_prx - a_best_prx

                        # Compute statistics on the difference in path pRx
                        prx_pathi_perc_i = np.percentile(path_rel_prx[:,path_i-1], REL_PATH_GAIN_PERCENTILES, interpolation='lower')
                        print("\tPRX gain (Path #1 - #{}) (DFT beams): Test {}".format(path_i+1, self.DATA_TAGS[snr_ind]))
                        print("\t\tMean: {}; Std-dev: {}".format(np.mean(path_rel_prx[:,path_i-1]), np.std(path_rel_prx[:,path_i-1])))
                        print("\t\tPercentiles:")
                        print(np.vstack((REL_PATH_GAIN_PERCENTILES, prx_pathi_perc_i)).T)
                        
                    else:
                        overall_best_prx = a_best_prx

                    prev_best_prx = a_best_prx

                    # Setup for the next path
                    # Determine the distance of the best beams to the next remaining beams
                    a_best_dist = np.abs((a_s_idx.T - a_best_idx.T).T)
                    # Determine which beams are too close to the current best beam
                    a_nwhere = np.nonzero(a_best_dist < MIN_PATH_SEPARATION)
                    
                    # Blackout the beams that are too close to the current best beam - avoid selecting these on the next iteration
                    for row_i in np.arange(prx_remain.shape[0]):
                        rm_idx_idx = a_nwhere[1][a_nwhere[0] == row_i]
                        rm_idx = a_s_idx[row_i, rm_idx_idx]
                        prx_remain[row_i, rm_idx] = DEFAULT_MIN_VAL

                # prx_dft_all_sorted = np.sort(prx_dft_all_date_i,axis=1)
                # prx_dft_all_sorted_idx = np.argsort(prx_dft_all_date_i,axis=1)
                # prx_dft_diff = prx_dft_all_sorted_idx 
                # for nlos_i in np.arange(NUM_NLOS):
                #     # Print out the relative gain (in dB) of each ranked DFT beam
                #     prx_pathi = prx_dft_all_sorted[:,-1] - prx_dft_all_sorted[:,-2-nlos_i]
                #     prx_pathi_perc_i = np.percentile(prx_pathi, REL_PATH_GAIN_PERCENTILES, interpolation='lower')
                #     print("\tPRX gain (Path #1 - #{}) (DFT beams): SNR {}".format(nlos_i+2, SNR_i))
                #     print("\t\tMean: {}; Std-dev: {}".format(np.mean(prx_pathi), np.std(prx_pathi)))
                #     print("\t\tPercentiles:")
                #     print(np.vstack((REL_PATH_GAIN_PERCENTILES, prx_pathi_perc_i)).T)
            
            # Plot STF-SNR vs. pRx (DFT labels only)
            plt.figure()
            plt.scatter(prx_dft_date_i, stfsnr_dft_date_i)
            plt.title("STF-SNR vs pRx for DFT Labels, Date: {}".format(self.DATA_DATES[snr_ind]))
            plt.xlabel("pRx (dB)")
            plt.ylabel("STF-SNR (dB)")
            
            print("\n\n")
                
        ## Compute the statistics - over all beams as well as individually
        # print("Computing statistics...")
        # for date_i in np.arange(NUM_DATES):
        #     stfsnr_dft_date_i = np.array([])
        #     sftsnr_pn_date_i = np.array([])
            
        #     for dft_i in np.arange(len(dft_use)):

