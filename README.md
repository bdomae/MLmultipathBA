# Multipath Machine Learning Assisted Beam Alignment Code and Data Repository
Publicly released software for the multipath machine learning assisted beam alignment algorithms and experiments published in Globecom 2021

See the pre-print article on arXiv: https://arxiv.org/abs/2109.14689


### Directory Structure - Highlights of the Important Files
data - Note that different dates refer to different capture times (with a different TX power and thus different SNR)
-> nn_sim* - Simulation data used to compare the algorithms in an ideal setting
-> dft64.csv - AWVs used for the directional exhaustive search codebook (64 beams)
-> pn36.csv - AWVs used for the compressive/PN beam codebook (36 beams)
-> results_awv0_<date>_<avg,ext>_dft.csv - RSS data for each of the 64 DFT beams (averaged or extended capture mode)
-> results_awv0_<date>_<avg,ext>_labels.csv - Post-processing selected best DFT beam index (label for training; averaged or extended capture mode)
-> results_awv0_<date>_<avg,ext>_pn.csv - RSS data for each of the 36 PN beams (averaged or extended capture mode)
-> results_awv0_<date>_<avg,ext>_stfsnr_<dft, labels, pn>.csv - STF-SNR data for the given collection methods
-> results_awv0_<date>_part*_<avg,ext>.csv - Raw data from the experimental capture (split up into consecutive parts for easier data capture)
non_coherent_matching_pursuit - MATLAB simulations for the RSS-MP method
tf0 - Machine learning Python (Jupyter) notebooks
-> all_NN_results_exp_train*rand.pkl - Saved results from the experimental data
-> all_NN_sim*_results.pkl - Saved results from the simulation data
-> mlbf_fcnet_sim*.ipynb - Jupyter notebooks for the neural network training and testing with simulated data
-> mlfb_fcnet3_real1<_varNum>.ipynb - Jupyter notebooks for the neural network training and testing with the experimental data
Array_calibration.m - CS algorithms with dictionary calibration MATLAB simulation
pattern_mismatch<_quan>.m - Antenna pattern mismatch evaluation due to hardware impairment MATLAB simulation
pRX.. .m - RSS-based beam training with various methods compared (MATLAB simulations)


### See also
mmRAPID original code repo: https://github.com/bdomae/mmRAPID
mmRAPID original paper: https://doi.org/10.1145/3412060.3418432