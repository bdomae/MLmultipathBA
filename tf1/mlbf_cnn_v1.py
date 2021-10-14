import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import mlbf_dataLoad_sim
import mlbf_dataLoad_exp
import mlbf_dataset
from mlbf_dataset import MLBF_Dataset, MLBF_Results


class mlbf_cnn:
    
    def __init__(self, NUM_CLASSES, NUM_MEAS, NUM_MEAS_COMBO, CHANNEL_TAG,
                 BATCH_SIZE, NUM_EPOCHS):
        #### Initialize the setup of the CNN

        ## Member constants
        self.NUM_CLASSES = NUM_CLASSES
        self.NUM_MEAS = NUM_MEAS
        self.NUM_MEAS_COMBO = NUM_MEAS_COMBO
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.MODEL_LABEL = "CNN"
        self.CHANNEL_TAG = CHANNEL_TAG

        ## Member variables
        self.all_models = []      # List to save the models (for each PN combo)
        self.all_loss = np.zeros((self.NUM_MEAS_COMBO, self.NUM_EPOCHS))
        self.all_train_acc = np.zeros((self.NUM_MEAS_COMBO, self.NUM_EPOCHS))
        self.all_results = MLBF_Results()


    def train(self, dataset, PLOT_PROGRESS=True):
        ### Setup and Train the NN Model ###

        ## Build the NN architecture
        # 6-layer CNN (3 conv1d + 3 fc layers)
        # ReLU activations
        # Sparse Categorical Cross-Entropy loss function

        # Architecture constants
        # fc_dims = [64, 128]  # Produced best results so far, best = 67-75% test acc
        # dropout_rate = 0.3
        train_dict_label = "ALL"
        dataset.convert_for_cnn()

        # Train a NN with each set of PN beams
        for pn_i in np.arange(self.NUM_MEAS_COMBO):
            print("\nTraining model for PN combo {}...".format(pn_i))
            
        #     # Test without dropout
        #     inputs = keras.Input(shape=(NUM_MEAS,), name='SparseRSSI')
        #     fc1 = layers.Dense(fc_dims[0], activation='relu', name='dense_1')(inputs)
        #     bn1 = layers.BatchNormalization()(fc1)
        #     #drp1 = layers.Dropout(dropout_rate)(bn1)
        #     fc2 = layers.Dense(fc_dims[1], activation='relu', name='dense_2')(bn1)
        #     bn2 = layers.BatchNormalization()(fc2)
        #     #drp2 = layers.Dropout(dropout_rate)(bn2)
        #     outputs = layers.Dense(NUM_CLASSES, name='predictions')(bn2)

        #     model = keras.Model(inputs=inputs, outputs=outputs)

            
            ## CNN architecture
            # 1D convolutions with maxpooling
            # batch norm to reduce initialization dependence
            dropped_units = 0.3
            model = keras.Sequential()

            model.add(layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=(self.NUM_MEAS, 1)))
            model.add(layers.MaxPooling1D(2, padding='same'))
            model.add(layers.Dropout(dropped_units))
            model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
            model.add(layers.MaxPooling1D(2, padding='same'))
            model.add(layers.Dropout(dropped_units))
            model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
            model.add(layers.Flatten())

            # model.add(layers.Dense(64, input_dim=NUM_MEAS))
            # model.add(layers.BatchNormalization())
            # model.add(layers.Activation('relu'))

            model.add(layers.Dense(64))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            #model.add(layers.Dropout(0.5))

            model.add(layers.Dense(64))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            #model.add(layers.Dropout(0.5))

            model.add(layers.Dense(self.NUM_CLASSES))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            #model.add(layers.Dropout(0.5))
            
            model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
                        # Loss function to minimize
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        # List of metrics to monitor
                        metrics=['sparse_categorical_accuracy'])

            # model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            #               loss='mse',       # mean squared error
            #               metrics=['mae'])  # mean absolute error

            # model.fit(train_data_all, train_labels_all, epochs=10, batch_size=NUM_MEAS,
            #           validation_data=(val_data_all, val_labels_all))
            
            # Train the NN
            loss_hist = model.fit(dataset.train_data_dict_r[train_dict_label][pn_i], dataset.train_classes_dict_r[train_dict_label][pn_i], epochs=self.NUM_EPOCHS, batch_size=self.BATCH_SIZE)
            
            # Save the data to the overall lists
            self.all_models.append(model)
            self.all_train_acc[pn_i, :] = loss_hist.history['sparse_categorical_accuracy']
            self.all_loss[pn_i, :] = loss_hist.history['loss']

        # Plot the results
        if PLOT_PROGRESS:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(self.all_train_acc.T)
            #plt.plot(loss_hist.history['val_acc'])
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            #plt.legend(["Training", "Validation"])

            plt.subplot(1,2,2)
            plt.plot(self.all_loss.T)
            #plt.plot(loss_hist.history['val_loss'])
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            #plt.legend(["Training", "Validation"])

        return self


    def test(self, dataset, 
             PRINT_RESULTS=True, PLOT_RESULTS=False,  
             PLOT_CONFUSION_MATRICES=False, SAVE_CONFUSION_MATRICES=False):
        #### Test the trained CNNs

        ## Pull the required constants from the configuration variable
        DATA_SNR = dataset.Config.DATA_SNR
        NUM_PN_COMBO = dataset.Config.NUM_PN_COMBO
        GAIN_LOSS_PERCENTILES = mlbf_dataset.GAIN_LOSS_PERCENTILES

        ### Test the NN Model ###
        class_test_acc = {}
        class_confusionMat = {}
        all_curMeas_acc = np.zeros((self.NUM_MEAS_COMBO, len(DATA_SNR)+1))         ####################
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

            for snr_ind in np.arange(len(DATA_SNR)+1):         ####################
                snr_tag = snr_ind
                if snr_ind != len(DATA_SNR):
                    SNR_i = DATA_SNR[snr_ind]
                    print("{} dB SNR --".format(SNR_i))
                else:
                    SNR_i = 'ALL'
                    snr_tag = 'ALL'
                    print("ALL SNR values --")
        #     for snr_ind in np.arange(len(DATA_SNR)):
        #         SNR_i = DATA_SNR[snr_ind]
        #         print("{} dB SNR --".format(SNR_i))
                test_predictions = self.all_models[pn_i].predict(dataset.test_data_dict_r[snr_tag][pn_i])
                print(test_predictions.shape)
                print('\tlabels:      ({}, {})'.format(np.min(dataset.test_classes_dict_r[snr_tag][pn_i]), np.max(dataset.test_classes_dict_r[snr_tag][pn_i])))
                print('\tpredictions: ({}, {})'.format(np.min(np.argmax(test_predictions, 1)), np.max(np.argmax(test_predictions, 1))))
                #print(val_labels[item,:])

                test_loss, test_acc = self.all_models[pn_i].evaluate(dataset.test_data_dict_r[snr_tag][pn_i],  dataset.test_classes_dict_r[snr_tag][pn_i], verbose=2)
                print('\tTest accuracy:', test_acc)
                class_test_acc[snr_tag] = test_acc

                # Compute the predicted labels and the confusion matrix
                test_pred_classes = np.argmax(test_predictions, 1)
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
                label_dft_ind = dataset.Config.dft_use[dataset.test_classes_dict_r[snr_tag][pn_i]].flatten()
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


