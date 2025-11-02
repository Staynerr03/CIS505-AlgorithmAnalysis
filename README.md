README FILE: 

This repository has the purpose of give examples of implementations of various anomaly detection algorithms:
SVM = detects unseen anomalous data using only normal training data
LOF = uses density and neighborhood relationships
LSTM Autoencoder = detects anomalies in time-series data
Graph-based LOF = detects suspicious nodes in networked data

FILES: 
1. Support Vector Machine for Novelty Detection- File: 01-Support Vector Machine for Novelty Detection.py

2. Local Outlier Factor (LOF)- File: 02-Local Outlier Factor (LOF).py

3. Time-Series Deep Learning Model (LSTM Autoencoder)- File: 03-Time-Series Deep Learning (LSTM Autoencoder for Reconstruction Anomaly Score).py

4. Graph-Based Detection Using Structural Features + LOF- File: 04-Graph-Based Detection (Structural Features + LOF).py


HOW TO RUN:  
    1. Clone the repository
    git clone https://github.com/Staynerr03/CIS505-AlgorithmAnalysis

    2. Install required packages
    pip install numpy scikit-learn networkx torch

How to Run Each Script:
    Support Vector Machine
    python "01-Support Vector Machine for Novelty Detection.py"

    Local Outlier Factor (LOF)
    python "02-Local Outlier Factor (LOF).py"

    LSTM Autoencoder for Time-Series
    python "03-Time-Series Deep Learning (LSTM Autoencoder for Reconstruction Anomaly Score).py"

    Graph-Based Detection
    python "04-Graph-Based Detection (Structural Features + LOF).py"
