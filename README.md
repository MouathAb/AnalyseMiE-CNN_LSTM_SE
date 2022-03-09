# AnalyseMiE-CNN_LSTM_SE



###########################################################################################

""" Packages to install """"

conda env create -f environment.yml --force

###########################################################################################

Insatll CASME II and SAMM data

Run python crop_6_ROIs.py   # Crop the Face into 6 ROIs

Run python CNN_SE.py        # spatial features extractions

Run python LSTM_SE.py       # spatio-temporal features extraction and classification

###########################################################################################

Note: the codes are initially set for evaluation of the proposed model with 9 patches for 5-AU classification using LOSO protocol on the mixed dataset of CASMEII and SAMM .
