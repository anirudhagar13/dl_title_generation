# file to store common use modules and imports
import re
import os
import sys
import random
import time
import datetime
import pickle
import json
import numpy as np
import pandas as pd

# pytorch modules
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# constants
SOS_token = 0
EOS_token = 1

dir_path = os.getcwd()
DATA_PATH = dir_path + '/main/data/training/abstitle_data.json'
ABS_LANG_PATH = dir_path + '/main/data/models/abs_lang.pkl'
TITLE_LANG_PATH = dir_path + '/main/data/models/title_lang.pkl'
ENC_MODEL_PATH = dir_path + '/main/data/models/encoder.pt'
DEC_MODEL_PATH = dir_path + '/main/data/models/attn_decoder.pt'

LR = 0.01
N_ITER = 40000
DROPOUT = 0.01
HIDDEN_SIZE = 512 
validate_num = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ('GPU availability: ', device)

def save_index(lang_index, path):
	'''
	simply pickles (serializes) embeddings / index
	'''
	with open(path, 'wb') as f:
		pickle.dump(lang_index, f)

def load_index(path):
	'''
	simply loads pickled (serialized) embeddings / index
	'''
	lang_index = dict()
	with open(path, 'rb') as f:
		lang_index = pickle.load(f)

	return lang_index

def save_model(model, path):
	'''
	just saves configurations (weights) of trained model
	'''
	torch.save(model.state_dict(), path)

def load_model(model, path):
	'''
	just loads configurations (weights) of trained model to recreate model
	'''
	if device == 'cuda':
		model.load_state_dict(torch.load(path))
	else:
		model.load_state_dict(torch.load(path, 
								map_location=torch.device('cpu')))
	return model

def load_data(path):
	'''
	returns data as a dataframe of tilte and abstract, irrespective of source
	'''
	data = list()

	if '.json' in path:
		with open(path) as f:
			data = json.load(f)
			data = pd.DataFrame(data)
	elif '.csv' in path:
		data = pd.read_csv(path)
		data = data[data['abstract'] != 'Abstract Missing'][['title','abstract']]
	else:
		raise Exception('Invalid data source')

	return data


