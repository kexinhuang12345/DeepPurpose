import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

import os

from DeepPurpose.utils import *
from DeepPurpose.model_helper import Encoder_MultipleLayers, Embeddings    
from DeepPurpose.encoders import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Sequential):
	def __init__(self, model_protein, **config):
		super(Classifier, self).__init__()
		self.input_dim_protein = config['hidden_dim_protein']

		self.model_protein = model_protein

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = config['cls_hidden_dims']
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim_protein] + self.hidden_dims + [1]
		
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v_P):
		# each encoding
		v_f = self.model_protein(v_P)
		# concatenate and classify
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f

def model_initialize(**config):
	model = Protein_Prediction(**config)
	return model

def model_pretrained(path_dir = None, model = None):
	if model is not None:
		path_dir = download_pretrained_model(model)
	config = load_dict(path_dir)
	model = Protein_Prediction(**config)
	model.load_pretrained(path_dir + '/model.pt')    
	return model

class Protein_Prediction:
	'''
		Protein Function Prediction 
	'''

	def __init__(self, **config):
		target_encoding = config['target_encoding']

		if target_encoding == 'AAC' or target_encoding == 'PseudoAAC' or  target_encoding == 'Conjoint_triad' or target_encoding == 'Quasi-seq' or target_encoding == 'ESPF':
			self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'], config['mlp_hidden_dims_target'])
		elif target_encoding == 'CNN':
			self.model_protein = CNN('protein', **config)
		elif target_encoding == 'CNN_RNN':
			self.model_protein = CNN_RNN('protein', **config)
		elif target_encoding == 'Transformer':
			self.model_protein = transformer('protein', **config)
		else:
			raise AttributeError('Please use one of the available encoding method.')

		self.model = Classifier(self.model_protein, **config)
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.target_encoding = target_encoding
		self.result_folder = config['result_folder']
		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)            
		self.binary = False
		if 'num_workers' not in self.config.keys():
			self.config['num_workers'] = 0
		if 'decay' not in self.config.keys():
			self.config['decay'] = 0

	def test_(self, data_generator, model, repurposing_mode = False, test = False, verbose = True):
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_p, label) in enumerate(data_generator):
			if self.target_encoding == 'Transformer':
				v_p = v_p
			else:
				v_p = v_p.float().to(self.device)              
			score = self.model(v_p)

			if self.binary:
				m = torch.nn.Sigmoid()
				logits = torch.squeeze(m(score)).detach().cpu().numpy()
			else:
				logits = torch.squeeze(score).detach().cpu().numpy()

			label_ids = label.to('cpu').numpy()
			y_label = y_label + label_ids.flatten().tolist()
			y_pred = y_pred + logits.flatten().tolist()
			outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
		
		model.train()
		if self.binary:
			if repurposing_mode:
				return y_pred
			## ROC-AUC curve
			if test:
				if verbose:
					roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
					plt.figure(0)
					roc_curve(y_pred, y_label, roc_auc_file, self.target_encoding)
					plt.figure(1)
					pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
					prauc_curve(y_pred, y_label, pr_auc_file, self.target_encoding)

			return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred
		else:
			if repurposing_mode:
				return y_pred
			return mean_squared_error(y_label, y_pred), \
				   pearsonr(y_label, y_pred)[0], \
				   pearsonr(y_label, y_pred)[1], \
				   concordance_index(y_label, y_pred), y_pred

	def train(self, train, val, test = None, verbose = True):
		if len(train.Label.unique()) == 2:
			self.binary = True
			self.config['binary'] = True

		lr = self.config['LR']
		decay = self.config['decay']

		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		if 'test_every_X_epoch' in self.config.keys():
			test_every_X_epoch = self.config['test_every_X_epoch']
		else:     
			test_every_X_epoch = 40
		loss_history = []

		self.model = self.model.to(self.device)

		# support multiple GPUs
		if torch.cuda.device_count() > 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
			self.model = nn.DataParallel(self.model, dim = 0)
		elif torch.cuda.device_count() == 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
		else:
			if verbose:
				print("Let's use CPU/s!")
		# Future TODO: support multiple optimizers with parameters
		opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)

		if verbose:
			print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': self.config['num_workers'],
	    		'drop_last': False}
		
		training_generator = data.DataLoader(data_process_loader_Protein_Prediction(train.index.values, 
																					 train.Label.values, 
																					 train, **self.config), 
																						**params)
		validation_generator = data.DataLoader(data_process_loader_Protein_Prediction(val.index.values, 
																						val.Label.values, 
																						val, **self.config), 
																						**params)
		
		if test is not None:
			info = data_process_loader_Protein_Prediction(test.index.values, test.Label.values, test, **self.config)
			params_test = {'batch_size': BATCH_SIZE,
					'shuffle': False,
					'num_workers': self.config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(info)}
			testing_generator = data.DataLoader(data_process_loader_Protein_Prediction(test.index.values, test.Label.values, test, **self.config), **params_test)

		# early stopping
		if self.binary:
			max_auc = 0
		else:
			max_MSE = 10000
		model_max = copy.deepcopy(self.model)

		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		if self.binary:
			valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
		else:
			valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
		table = PrettyTable(valid_metric_header)
		float2str = lambda x:'%0.4f'%x

		if verbose:
			print('--- Go for Training ---')
		t_start = time() 
		for epo in range(train_epoch):
			for i, (v_p, label) in enumerate(training_generator):
				
				if self.target_encoding == 'Transformer':
					v_p = v_p
				else:
					v_p = v_p.float().to(self.device) 

				score = self.model(v_p)
				label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

				if self.binary:
					loss_fct = torch.nn.BCELoss()
					m = torch.nn.Sigmoid()
					n = torch.squeeze(m(score), 1)
					loss = loss_fct(n, label)
				else:
					loss_fct = torch.nn.MSELoss()
					n = torch.squeeze(score, 1)
					loss = loss_fct(n, label)
				loss_history.append(loss.item())

				opt.zero_grad()
				loss.backward()
				opt.step()

				if verbose:
					if (i % 100 == 0):
						t_now = time()
						if verbose:
							print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
							' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
							". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
						### record total run time

			##### validate, select the best model up to now 
			with torch.set_grad_enabled(False):
				if self.binary:  
					## binary: ROC-AUC, PR-AUC, F1  
					auc, auprc, f1, logits = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
					valid_metric_record.append(lst)
					if auc > max_auc:
						model_max = copy.deepcopy(self.model)
						max_auc = auc
					if verbose:
						print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
						  ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7])
				else:  
					### regression: MSE, Pearson Correlation, with p-value, Concordance Index  
					mse, r2, p_val, CI, logits = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
					valid_metric_record.append(lst)
					if mse < max_MSE:
						model_max = copy.deepcopy(self.model)
						max_MSE = mse
					if verbose:
						print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
						 + str(r2)[:7] + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI)[:7])
			table.add_row(lst)


		#### after training 
		prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(table.get_string())

		# load early stopped model
		self.model = model_max

		if test is not None:
			if verbose:
				print('--- Go for Testing ---')
			if self.binary:
				auc, auprc, f1, logits = self.test_(testing_generator, model_max, test = True, verbose = verbose)
				test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
				test_table.add_row(list(map(float2str, [auc, auprc, f1])))
				if verbose:
					print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))				
			else:
				mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max, test = True, verbose = verbose)
				test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
				test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
				if verbose:
					print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
					  + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
			np.save(os.path.join(self.result_folder, str(self.target_encoding)
				     + '_logits.npy'), np.array(logits))                

			######### learning record ###########

			### 1. test results
			prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
			with open(prettytable_file, 'w') as fp:
				fp.write(test_table.get_string())

		if verbose:
		### 2. learning curve 
			fontsize = 16
			iter_num = list(range(1,len(loss_history)+1))
			plt.figure(3)
			plt.plot(iter_num, loss_history, "bo-")
			plt.xlabel("iteration", fontsize = fontsize)
			plt.ylabel("loss value", fontsize = fontsize)
			pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
			with open(pkl_file, 'wb') as pck:
				pickle.dump(loss_history, pck)

			fig_file = os.path.join(self.result_folder, "loss_curve.png")
			plt.savefig(fig_file)
		if verbose:
			print('--- Training Finished ---')
          

	def predict(self, df_data, verbose = True):
		'''
			utils.data_process_repurpose_virtual_screening 
			pd.DataFrame
		'''
		if verbose:
			print('predicting...')
		info = data_process_loader_Protein_Prediction(df_data.index.values, df_data.Label.values, df_data, **self.config)
		self.model.to(device)
		params = {'batch_size': self.config['batch_size'],
				'shuffle': False,
				'num_workers': self.config['num_workers'],
				'drop_last': False,
				'sampler':SequentialSampler(info)}

		generator = data.DataLoader(info, **params)

		score = self.test_(generator, self.model, repurposing_mode = True)
		# set repurposong mode to true, will return only the scores.
		return score

	def save_model(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(self.model.state_dict(), path_dir + '/model.pt')
		save_dict(path_dir, self.config)

	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		if self.device == 'cuda':
			state_dict = torch.load(path)
		else:
			state_dict = torch.load(path, map_location = torch.device('cpu'))
		# to support training from multi-gpus data-parallel:
        
		if next(iter(state_dict))[:7] == 'module.':
			# the pretrained model is from data-parallel module
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:] # remove `module.`
				new_state_dict[name] = v
			state_dict = new_state_dict

		self.model.load_state_dict(state_dict)

		self.binary = self.config['binary']


