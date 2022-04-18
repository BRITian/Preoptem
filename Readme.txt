# Preoptem

Introduction
====
**Preoptem** is an open-source tool for predicting the optimal temperature of proteins based on the deep learning methods.

System requirement
=====
1. Python 2.7
2. tensorflow 1.15.0
3. keras 2.1.5
4. theano 1.0.5
5. opencv-python 4.1.2.30


Quick Start to install the required program
=====
1. Install the python 2.7 (from Anaconda https://www.anaconda.com/)
2. pip install tensorflow==1.15.0 (python=2.7)
3. pip install keras==2.1.5
4. pip install theano==1.0.5
5. pip install opencv-python==4.1.2.30
6. pip install matplotlib==2.2.5
7. git clone https://github.com/BRITian/Preoptem

Predict the sequence 
====
The input sequences should be in the fasta format. The input command is as the followings.
python Preoptem_keras_RNN_pre.py test_chi.fas


Result analysis 
====
The output of the prediction is saved in the file of ./preoptem_pres/test_chi.fas.pred.
The values in the first, second and third column in the file represented the Seq_id, Predicted_optimal_temperature, and Predicted_class.
