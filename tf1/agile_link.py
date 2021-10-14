import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import mlbf_dataLoad_sim
import mlbf_dataLoad_exp
import mlbf_dataset
from mlbf_dataset import MLBF_Dataset, MLBF_Results


class agile_link:

    def __init__(self, NUM_CLASSES, NUM_MEAS, NUM_MEAS_COMBO, CHANNEL_TAG,
                 selected_beam_idxs, selected_dft_idxs, beam_type='dftsa', 
                 codebook_file='../data/Codebooks/awv_dft64_pn36_qpd10_dftsa10.csv', ant_d=0.57):
        #### Initialize the setup of the Agile-Link algorithm
        #       - selected_beam_idxs = Dictionary indexed by measurement index
        #       - selected_dft_idxs = np.array of all DFT beams used as labels
        #       - beam_type -> Beam type codes: 'dft' = DFT beams; 'cs' == PN beams; 'qpd' = QPD widened beams; 'dftsa' = SA multifinger beams

        ## Member constants
        self.NUM_CLASSES = NUM_CLASSES
        self.NUM_MEAS = NUM_MEAS
        self.NUM_MEAS_COMBO = NUM_MEAS_COMBO
        self.CHANNEL_TAG = CHANNEL_TAG
        if beam_type == 'dftsa':
            self.MODEL_LABEL = "AgileLink"
        elif beam_type == 'cs':
            self.MODEL_LABEL = "CS-RSS-MP"
        self.ANT_D = ant_d

        ## Member variables
        self.all_models = []      # List to save the models (for each PN combo)
        # self.all_loss = np.zeros((self.NUM_MEAS_COMBO, 1))
        self.all_train_acc = np.zeros((self.NUM_MEAS_COMBO, 1))
        self.all_results = MLBF_Results()
        self.codebook_file = codebook_file

        ## Load the codebook file
        self.beam_type = beam_type 
        self.df_codebooks = pd.read_csv(codebook_file, skipinitialspace=True)

        # Finding the AWVs for the measurements (i.e. the sensing codebook)
        cols_ant = self.df_codebooks.keys().str.contains('Ant')
        self.Nr = np.sum(cols_ant)
        self.codebook_phase = self.df_codebooks.loc[self.df_codebooks['BeamLabel'].str.contains(self.beam_type)].loc[:,cols_ant].to_numpy()         # AWVs in the rows (i.e. codebook dim is (# of meas) x (# antennas))
        self.codebook_all = np.exp(1j*np.radians(self.codebook_phase)).T                                                                            # Dim: (# antennas) x (# of meas)

        # Finding the angles represented by each DFT beam (used in pattern estimation)
        col_angle = self.df_codebooks.keys().str.contains("Angle")
        self.dft_angles_all = self.df_codebooks.loc[self.df_codebooks['BeamLabel'].str.contains("dft[0-9]")].loc[:,col_angle].to_numpy().flatten()  # Angles for the DFT beams (in order by index)

        # Store the system measurement parameters (i.e. the beam indices used)
        self.meas_idxs = selected_beam_idxs     # Beam indices for the DFTSA beams for measurements
        self.dft_idxs = selected_dft_idxs       # Beam indices for the DFT beams used in the dataset
        self.dft_angles = self.dft_angles_all[self.dft_idxs]
        

    def train(self, dataset,  PLOT_PROGRESS=False):
        #### Training for Agile-Link is just approximating our multifinger beam patterns using their theoretical designs
        #       - Wanted to calibrate the beam patterns with the training data, 
        #         but that is non-trivial/may not be possible for NLOS/multipath data

        # Additional temporary variables used to compute the patterns
        n = np.arange(self.Nr)
        if (self.beam_type == 'cs'):
            exp = (1/np.sqrt(self.Nr))*np.exp(1j*n*2*np.pi*self.ANT_D*np.array([np.sin(np.deg2rad(self.dft_angles))]).T).T     # Dim: (# antennas) x (# DFT labels used)
        else:
            exp = (1/np.sqrt(self.Nr))*np.exp(-1j*n*2*np.pi*self.ANT_D*np.array([np.sin(np.deg2rad(self.dft_angles))]).T).T     # Dim: (# antennas) x (# DFT labels used)

        # Initialize the dictionaries for the pattern information
        self.codebook = {}
        self.pattern_weights = {}

        for pn_i in np.arange(self.NUM_MEAS_COMBO):
            # Find the beams that are actually part of the codebook
            self.codebook[pn_i] = self.codebook_all[:,self.meas_idxs[pn_i,:]]       # Dim: (# antennas) x (# meas) 
            
            # Estimate the beam patterns (right now just using the theoretical values; maybe a way to do a full estimation, but that's nontrivial)
            raw_array_factor = np.abs(np.matmul(self.codebook[pn_i].conj().T, exp))
            norm_array_factor = (raw_array_factor/np.linalg.norm(raw_array_factor, ord=2, axis=0)) # Normalized over the measurement dimension (each DFT label result is normalized)
            # self.pattern_weights[pn_i] = np.square(np.abs(norm_array_factor))       # Dim: (# meas) x (# DFT labels used)
            self.pattern_weights[pn_i] = norm_array_factor                          # Dim: (# meas) x (# DFT labels used)
        #import pdb; pdb.set_trace()


    def predict(self, test_data_vec, pn_i, out_scores=True):
        #### Function to predict the result for a given test vector (since no model class is used)
        # num_test_pts = test_data_vec.shape[0]           #TODO: CHECK DIMS

        # Compute score
        norm_test_data_vec = (test_data_vec.T/np.linalg.norm(test_data_vec, ord=2, axis=1)).T
        all_scores = np.matmul(norm_test_data_vec, self.pattern_weights[pn_i])   # Should be Dim: (# points) x (# DFT labels used)

        # Find the max score -> specify the best beam
        all_predictions = np.argmax(all_scores, axis=1)     # Should be Dim: (# points) x 1

        if out_scores:
            return (all_predictions, all_scores)
        return all_predictions
        #import pdb; pdb.set_trace()


    def test(self, dataset,
             PRINT_RESULTS=True, PLOT_RESULTS=False,  
             PLOT_CONFUSION_MATRICES=False, SAVE_CONFUSION_MATRICES=False):
        #### Test the Agile-Link algorithm

        ## Pull the required constants from the configuration variable
        DATA_SNR = dataset.Config.DATA_SNR
        NUM_PN_COMBO = dataset.Config.NUM_PN_COMBO
        GAIN_LOSS_PERCENTILES = mlbf_dataset.GAIN_LOSS_PERCENTILES

        ### Test the NN Model ###
        class_test_acc = {}
        class_confusionMat = {}
        all_curMeas_acc = np.zeros((self.NUM_MEAS_COMBO, len(DATA_SNR)+1))
        all_curMeas_test_confusion = np.zeros((self.NUM_MEAS_COMBO, len(DATA_SNR)+1, self.NUM_CLASSES, self.NUM_CLASSES))
        all_curMeas_pe = np.zeros((self.NUM_MEAS_COMBO, len(DATA_SNR)+1, self.NUM_CLASSES))
        all_curMeas_gainloss = {}   # Format: {pn_i}{SNR_i}[test_trial] ({} = dict, [] = np array) 

        for pn_i in np.arange(self.NUM_MEAS_COMBO):
            print("\nPN combo {}...".format(pn_i))
            pn_i_gainloss = {} 
            
            # Note: SNR values and indices are used in the following ways
            #   1) SNR_i = the actual value/name of the data's SNR (i.e. 20 (db) or "ALL")
            #       Used for labels
            #   2) snr_ind = the index of the SNR value (i.e. 0,...,(NUM_SNR-1)) 
            #       Used for Numpy arrays (basically only results)
            #   3) snr_tag = "Tag" of the SNR (i.e. snr_ind except for the "ALL" case)
            #       Used as a dictionary key (for datasets and odd sized results)

            for snr_ind in np.arange(len(DATA_SNR)+1):
                snr_tag = snr_ind
                if snr_ind != len(DATA_SNR):
                    SNR_i = DATA_SNR[snr_ind]
                    print("{} dB SNR --".format(SNR_i))
                else:
                    SNR_i = 'ALL'
                    snr_tag = 'ALL'
                    print("ALL SNR values --") 

                test_pred_classes, test_predictions = self.predict(dataset.test_data_dict[snr_tag][pn_i], pn_i)
                print(test_predictions.shape)
                print('\tlabels:      ({}, {})'.format(np.min(dataset.test_classes_dict[snr_tag][pn_i]), np.max(dataset.test_classes_dict[snr_tag][pn_i])))
                print('\tpredictions: ({}, {})'.format(np.min(np.argmax(test_predictions, 1)), np.max(np.argmax(test_predictions, 1))))
                #print(val_labels[item,:])

                test_acc = np.sum(test_pred_classes == dataset.test_classes_dict[snr_tag][pn_i].flatten())/(test_pred_classes.shape[0])
                print('\tTest accuracy:', test_acc)
                class_test_acc[snr_tag] = test_acc
                #import pdb; pdb.set_trace()

                # Compute the predicted labels and the confusion matrix
                test_confusion = tf.math.confusion_matrix(dataset.test_classes_dict[snr_tag][pn_i], #non r?
                                                        test_pred_classes)
                #print(test_confusion)
                class_confusionMat[snr_tag] = test_confusion
                CLASSES = dataset.Config.dft_use

                ## Plot the confusion matrix (see which beams get associated with each other)
                if PLOT_CONFUSION_MATRICES:
                    fig, ax = plt.subplots()
                    ax.matshow(test_confusion)
                    for (i, j), z in np.ndenumerate(test_confusion):
                        ax.text(j, i, '{:d}'.format(z), ha='center', va='center')

                    ax.set_xticklabels(CLASSES)
                    ax.set_yticklabels(CLASSES)
                    ax.xaxis.set_label_position("top")
                    fig.set_size_inches(12, 12)
                    plt.title("Beamtraining Confusion Matrix ({} SNR)".format(SNR_i))
                    plt.xlabel("Predicted angles (degrees)")
                    plt.ylabel("True angles (degrees)")
                    plt.xticks(np.arange(self.NUM_CLASSES))
                    plt.yticks(np.arange(self.NUM_CLASSES))
                    if SAVE_CONFUSION_MATRICES:
                        plt.savefig('../figures/confusionMatrix_sim4_{}meas_{}SNR.png'.format(self.NUM_MEAS, SNR_i))
                    plt.show()

                ## Plot the P(e) given a specific angle
                num_true_angles = np.sum(test_confusion,1)
                num_correct = np.diag(test_confusion)
                num_incorrect = num_true_angles - num_correct
                pe = num_incorrect/num_true_angles

                ## Compute the gain loss for each test beam
                label_dft_ind = dataset.Config.dft_use[dataset.test_classes_dict[snr_tag][pn_i]].flatten()
                max_gain = dataset.dft_rssi_dict[snr_tag][pn_i][np.arange(len(label_dft_ind)), label_dft_ind]
                selected_dft_ind = dataset.Config.dft_use[test_pred_classes]
                achieved_gain = dataset.dft_rssi_dict[snr_tag][pn_i][np.arange(len(selected_dft_ind)), selected_dft_ind]
                pn_i_gainloss[snr_tag] = max_gain - achieved_gain

                ## Store the results for a final overall plot
                all_curMeas_acc[pn_i, snr_ind] = test_acc
                all_curMeas_test_confusion[pn_i, snr_ind, :, :] = test_confusion
                all_curMeas_pe[pn_i, snr_ind, :] = pe

            ## Store the gain loss dictionary for this PN beam combo
            all_curMeas_gainloss[pn_i] = pn_i_gainloss
                
        ## Post-process the results and combine for plots
        print("\nNumber of results:            {}".format(all_curMeas_acc.shape))
        print("Confusion matrix tensor size: {}".format(all_curMeas_test_confusion.shape))

        # Plot the P(e) by DFT label for each SNR         ####################
        if PLOT_RESULTS:
            colors_snr = ['b', 'g', 'm', 'c', 'r']
            for snr_ind in np.arange(len(DATA_SNR)+1):
                if snr_ind != len(DATA_SNR):
                    SNR_i = DATA_SNR[snr_ind]
                    print("{} dB SNR --".format(SNR_i))
                else:
                    SNR_i = 'ALL'
                fig = plt.figure()
                fig.set_size_inches(10, 4)
                plt.plot(np.arange(self.NUM_CLASSES), all_curMeas_pe[:,snr_ind,:].T, color=colors_snr[snr_ind], linewidth=0.5)
                plt.plot(np.arange(self.NUM_CLASSES), np.mean(all_curMeas_pe, 0)[snr_ind], color=colors_snr[snr_ind], linewidth=5)
                plt.title("Classification Error by True Angle ({} SNR)".format(SNR_i))
                plt.xlabel("DFT True Beam Index")
                plt.ylabel("P(e)")
                plt.xticks(np.arange(self.NUM_CLASSES), CLASSES, rotation=90)

        # Plot the gain loss peformance
        all_gainloss_perc = np.zeros((len(DATA_SNR)+1, NUM_PN_COMBO, len(GAIN_LOSS_PERCENTILES)))         ####################
        for snr_ind in np.arange(len(DATA_SNR)+1):
            snr_tag = snr_ind
            if snr_ind != len(DATA_SNR):
                SNR_i = DATA_SNR[snr_ind]
                print("{} dB SNR --".format(SNR_i))
            else:
                SNR_i = 'ALL'
                snr_tag = 'ALL'
                print("ALL SNR values --")
        #     SNR_i = DATA_SNR[snr_ind]
            # Plot the results
            if PLOT_RESULTS:
                fig = plt.figure()
                for pn_i in np.arange(NUM_PN_COMBO):
                    all_gainloss_perc[snr_ind, pn_i, :] = np.percentile(all_curMeas_gainloss[pn_i][snr_tag], GAIN_LOSS_PERCENTILES, interpolation='lower')
                    plt.plot(GAIN_LOSS_PERCENTILES, all_gainloss_perc[snr_ind, pn_i, :].T, label=dataset.Config.codebook_labels[pn_i])
                fig.set_size_inches(6, 4)
                plt.title("Gain Loss Percentiles for SNR={}".format(SNR_i))
                plt.legend()
                plt.xlabel("Percentile")
                plt.ylabel("Gain Loss (dB)")
            else:
                for pn_i in np.arange(NUM_PN_COMBO):
                    all_gainloss_perc[snr_ind, pn_i, :] = np.percentile(all_curMeas_gainloss[pn_i][snr_tag], GAIN_LOSS_PERCENTILES, interpolation='lower')
            if PRINT_RESULTS:
                print("Gain Loss Percentiles, SNR {}".format(SNR_i))
                print(np.vstack((GAIN_LOSS_PERCENTILES, all_gainloss_perc[snr_ind, :, :])).T)
            
        # Plot the test accuracy vs SNR (each PN beam combo and the overall averages)
        avg_acc_snr = np.mean(all_curMeas_acc, 0)
        if PLOT_RESULTS:
            fig = plt.figure()
            fig.set_size_inches(6, 4)
            plt.plot(DATA_SNR, all_curMeas_acc[:, np.arange(len(DATA_SNR))].T)                 # Plot all the PN beam combos         ####################
            plt.plot(DATA_SNR, avg_acc_snr[np.arange(len(DATA_SNR))], color='r', linewidth=5)  # Plot the average per SNR
            plt.title("Test Accuracy by SNR")
            plt.xlabel("SNR")
            plt.ylabel("Test Accuracy")

        print("\nAverage Test Accuracies: ")
        for snr_ind in np.arange(len(DATA_SNR)):
            print("\tSNR: {}; Accuracy: {}".format(DATA_SNR[snr_ind], avg_acc_snr[snr_ind]))

        # Save the results for this number of measurements to the overall dictionaries
        self.all_results.save_results(self.NUM_MEAS, self.MODEL_LABEL, self.CHANNEL_TAG,
                                      all_curMeas_acc, all_curMeas_test_confusion, 
                                      all_curMeas_pe, all_gainloss_perc, 
                                      dataset.Config.codebook_labels, dataset.Config.ALL_USE_BEAMS)

