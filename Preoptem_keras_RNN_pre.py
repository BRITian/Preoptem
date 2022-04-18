import os
#os.environ['KERAS_BACKEND']='theano'
import numpy as np
#import theano
import keras
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model
import shutil, os, sys
from numpy import genfromtxt
from keras.preprocessing import sequence
#theano.config.openmp = True
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning) 

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4, suppress=True)
import tensorflow as tf


#epochs = 20

from keras.layers.normalization import BatchNormalization
from group_norm import GroupNormalization

import keras.backend.tensorflow_backend as KTF
from keras.utils.generic_utils import CustomObjectScope
session_config = tf.compat.v1.ConfigProto(log_device_placement=True,inter_op_parallelism_threads=0,intra_op_parallelism_threads=0,allow_soft_placement=True)


script_start1 = dt.datetime.now()

in_name=sys.argv[1]
#in_model=sys.argv[2]
max_seq_len=1000
fold=10
time_thd = 360  # s

def name_seq(fasta_file):
	with open(fasta_file, 'r') as file:
		lines = file.readlines()
	name_list, seq_list = [], []
	for line in lines:
		if '>' in line:
			name_list.append(line.strip().replace('>', ''))
			seq_list.append('')
		else:
			seq_list[-1] = '%s%s' % (seq_list[-1], line.strip().upper())
	return name_list, seq_list

def coding_aa(aa_list):
	coding_list, miss_info = [], 0
	aac_coding = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
				'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
				'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
	for item in aa_list:
		if aac_coding.get(item) is None:
			coding_list.append('0')
			miss_info += 1
		else:
			coding_list.append(str(aac_coding.get(item)))
	if miss_info > 5:
		print('Please check your Amino acid sequence, there are many "gap" or "non-AA" letters.')
		sys.exit()
	return coding_list

def count_ave_std(res_list, step):
	temp_ave, temp_std = [], []
	for s1 in range(step):
		temp_data = res_list[s1::step]
		temp_ave.append(np.average(temp_data))
		temp_std.append(np.std(temp_data))
	return temp_ave, temp_std

pred_dir = './preoptem_pres'
if not os.path.exists(pred_dir):
	os.makedirs(pred_dir)

name_list, seq_list = name_seq(in_name)

file_error_name = '%s.error' % in_name
file_pred_name = '%s.pred' % in_name
file_coding_url = "./%s/%s_aac.txt" % (pred_dir,in_name)

#infile_train="./preoptem_pres/" + in_name

file_coding = open(file_coding_url, 'w', 0)
#print("seq_feature: %s => Seq_type: Prot" % seq_feature)
for name, seq in zip(name_list, seq_list):
	file_coding.write('0 %s ,%s ,\n' % (name.replace(' ', '-').replace(',', '*'), " ".join(coding_aa(seq))))
file_coding.close()

#sys.exit();

train_data = pd.read_csv(file_coding_url, index_col = False, header=None)
x_train_ori=train_data[1]
y_train_ori = train_data[0]

x_train=[]
for pi in x_train_ori:
	nr=pi.split(' ')[0:-1]
	ndata=map(int,nr)
	x_train.append(ndata)
x_train=np.array(x_train)
x_train=sequence.pad_sequences(x_train,maxlen=max_seq_len,padding='post',truncating='post')

file_pred_url = '%s/%s' % (pred_dir, file_pred_name)
file_pred = open(file_pred_url, 'w', 0)

all_result = []

for f1 in range(fold):

	model_url = 'models/Best_model_REG_R%s.h5' % (f1+1)

	#KTF.clear_session()
	#session = tf.Session(config=session_config)
	#KTF.set_session(session)
	#with CustomObjectScope({}):
	model = load_model(model_url)

	pred_result = model.predict(x_train)
	print('# === Fold: %s ===\n' % (f1 + 1))
	#print(pred_result)
	for i in range(len(pred_result)):
		all_result.append(pred_result[i])
	del model

avepmt, stdpmt = count_ave_std(all_result, len(y_train_ori))
file_pred.write('Seq_id\tPredicted_optimal_temperature\tPredicted_class\n')
for i in range(len(y_train_ori)):
	tar_pred=avepmt[i]*17.32803929+40.92039474
	if tar_pred >50:
		tar_pred_class="Thermophilic_protein"
	else:
		if tar_pred <20:
			tar_pred_class="Psychrophilic_protein"
		else:
			tar_pred_class="Mesophilic_protein"

	#tar_pred_class=stdpmt[i]*17.32803929+40.92039474
	file_pred.write('%s\t%.4f\t%s\n' % (name_list[i],tar_pred, tar_pred_class))
file_pred.close()

sys.exit()

