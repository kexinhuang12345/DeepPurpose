import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn 

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr


torch.manual_seed(2)    # reproducible torch:2 np:3
np.random.seed(3)
import copy

from utils import data_process_loader, data_process_repurpose_virtual_screening, create_var, index_select_ND




class MLP(nn.Sequential):
	def __init__(self, input_dim, hidden_dim, hidden_dims):
		super(MLP, self).__init__()
		layer_size = len(hidden_dims) + 1
		dims = [input_dim] + hidden_dims + [hidden_dim]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v):
		# predict
		for i, l in enumerate(self.predictor):
			v = l(v)
		return v  


class MPNN(nn.Sequential):

	def __init__(self, mpnn_hidden_size, mpnn_depth):
		super(MPNN, self).__init__()
		self.mpnn_hidden_size = mpnn_hidden_size
		self.mpnn_depth = mpnn_depth 
		from chemutils import ATOM_FDIM, BOND_FDIM

		self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
		self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
		self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

	def forward(self, fatoms, fbonds, agraph, bgraph):
		fatoms = create_var(fatoms)
		fbonds = create_var(fbonds)
		agraph = create_var(agraph)
		bgraph = create_var(bgraph)

		binput = self.W_i(fbonds)
		message = F.relu(binput)

		for i in range(self.mpnn_depth - 1):
			nei_message = index_select_ND(message, 0, bgraph)
			nei_message = nei_message.sum(dim=1)
			nei_message = self.W_h(nei_message)
			message = F.relu(binput + nei_message)

		nei_message = index_select_ND(message, 0, agraph)
		nei_message = nei_message.sum(dim=1)
		ainput = torch.cat([fatoms, nei_message], dim=1)
		atom_hiddens = F.relu(self.W_o(ainput))
		print(atom_hiddens.shape)








class Classifier(nn.Sequential):
	def __init__(self, model_drug, model_protein, **config):
		super(Classifier, self).__init__()
		self.input_dim_drug = config['hidden_dim_drug']
		self.input_dim_protein = config['hidden_dim_protein']

		self.model_drug = model_drug
		self.model_protein = model_protein

		self.hidden_dims = config['cls_hidden_dims']
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]
		
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v_D, v_P):
		# each encoding
		v_D = self.model_drug(v_D)
		v_P = self.model_protein(v_P)
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 1)

		for i, l in enumerate(self.predictor):
			v_f = l(v_f)

		return v_f    

def model_initialize(drug_encoding, target_encoding, **config):
	model = DBTA(drug_encoding, target_encoding, **config)
	return model

def model_pretrained(path, drug_encoding, target_encoding, **config):
	model = DBTA(drug_encoding, target_encoding, **config)
	model.load_pretrained(path)
	return model

def repurpose(X_repurpose, target, model, drug_names = None, target_name = None):
	# X_repurpose: a list of SMILES string
	# target: one target 

	print('repurposing...')
	X_repurpose, target = data_process_repurpose_virtual_screening(X_repurpose, target, model.drug_encoding, model.target_encoding, 'repurposing')
	y_pred = model.predict((X_repurpose, target))

	if target_name is not None:
		print('Drug Repurposing Result for '+target_name)
	if drug_names is not None:
		f_d = max([len(o) for o in drug_names]) + 1
		for i in range(X_repurpose.shape[0]):
			if model.binary:
				if y_pred[i].item() > 0.5:
					print('{:<{f_d}}'.format(drug_names[i], f_d =f_d) + ' predicted to have interaction with the target')	
				else:
					print('{:<{f_d}}'.format(drug_names[i], f_d =f_d) + ' predicted to NOT have interaction with the target')								
			else:
				print('{:<{f_d}}'.format(drug_names[i], f_d =f_d) + ' predicted to have binding affinity score ' + "{0:.2f}".format(y_pred[i].item()))

	return y_pred

def virtual_screening(X_repurpose, target, model, drug_names = None, target_names = None):
	# X_repurpose: a list of SMILES string
	# target: a list of targets
	print('repurposing...')
	X_repurpose, target = data_process_repurpose_virtual_screening(X_repurpose, target, model.drug_encoding, model.target_encoding, 'virtual screening')
	y_pred = model.predict((X_repurpose, target))

	if drug_names is not None and target_names is not None:
		print('Virtual Screening Result')
		f_d = max([len(o) for o in drug_names]) + 1
		f_p = max([len(o) for o in target_names]) + 1
		for i in range(X_repurpose.shape[0]):
			if model.binary:
				if y_pred[i].item() > 0.5:
					print('{:<{f_d}}'.format(drug_names[i], f_d =f_d) + ' predicted to have interaction with the target ' + '{:<{f_p}}'.format(target_names[i], f_p =f_p))	
				else:
					print('{:<{f_d}}'.format(drug_names[i], f_d =f_d) + ' predicted to NOT have interaction with the target ' + '{:<{f_p}}'.format(target_names[i], f_p =f_p))								
			else:
				print('{:<{f_d}}'.format(drug_names[i], f_d =f_d) + ' and target ' + '{:<{f_p}}'.format(target_names[i], f_p =f_p) + ' predicted to have binding affinity score ' + "{0:.2f}".format(y_pred[i].item()))
	return y_pred


class DBTA:
	def __init__(self, drug_encoding, target_encoding, **config):
		### merge fix a small bug  if or or or 
		if drug_encoding == 'Morgan' or drug_encoding=='Pubchem' or drug_encoding=='Daylight' or drug_encoding=='rdkit_2d_normalized':
			# Future TODO: support multiple encoding scheme for static input 
			self.model_drug = MLP(config['input_dim_drug'], config['hidden_dim_drug'], config['mlp_hidden_dims_drug'])
		elif drug_encoding == 'SMILES_CNN':
			raise NotImplementedError
		elif drug_encoding == 'SMILES_Transformer':
			raise NotImplementedError
		elif drug_encoding == 'MPNN':
			self.model_drug = MPNN(config['mpnn_hidden_size'], config['mpnn_depth'])
			#raise NotImplementedError
			#self.model_drug = MPNN()
		else:
			raise AttributeError('Please use one of the available encoding method.')

		if target_encoding == 'AAC' or target_encoding=='PseudoAAC' or target_encoding=='Conjoint_triad' or target_encoding=='Quasi-seq':
			self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'], config['mlp_hidden_dims_target'])
		elif target_encoding == 'CNN':
			raise NotImplementedError
		elif target_encoding == 'attention_CNN':
			raise NotImplementedError
		elif target_encoding == 'RNN':
			raise NotImplementedError			
		elif target_encoding == 'Transformer':
			raise NotImplementedError			
		else:
			raise AttributeError('Please use one of the available encoding method.')

		self.model = Classifier(self.model_drug, self.model_protein, **config)
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.drug_encoding = drug_encoding
		self.target_encoding = target_encoding
		self.binary = False

	def test_(self, data_generator, model):
	    y_pred = []
	    y_label = []
	    model.eval()
	    for i, (v_d, v_p, label) in enumerate(data_generator):
	        score = model(v_d.float().to(self.device), v_p.float().to(self.device))
	        if self.binary:
	        	m = torch.nn.Sigmoid()
	        	logits = torch.squeeze(m(score)).detach().cpu().numpy()
	        else:
		        logits = torch.squeeze(score).detach().cpu().numpy()
	        label_ids = label.to('cpu').numpy()
	        y_label = y_label + label_ids.flatten().tolist()
	        y_pred = y_pred + logits.flatten().tolist()
	        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
	    if self.binary:
	    	return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred
	    else:
		    return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred

	def train(self, train, val, test = None):
		if len(train.Label.unique()) == 2:
			self.binary = True

		lr = self.config['LR']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		loss_history = []

		self.model = self.model.to(self.device)

		# support multiple GPUs
		if torch.cuda.device_count() > 1:
			print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
			self.model = nn.DataParallel(self.model, dim = 0)

		# Future TODO: support multiple optimizers with parameters
		opt = torch.optim.Adam(self.model.parameters(), lr = lr)

		print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': 0,
	    		'drop_last': False}

		training_generator = data.DataLoader(data_process_loader(train.index.values, train.Label.values, train, **self.config), **params)
		validation_generator = data.DataLoader(data_process_loader(val.index.values, val.Label.values, val, **self.config), **params)
		
		if test is not None:
			testing_generator = data.DataLoader(data_process_loader(test.index.values, test.Label.values, test, **self.config), **params)

		# early stopping
		if self.binary:
			max_auc = 0
		else:
			max_MSE = 10000
		model_max = copy.deepcopy(self.model)

		print('--- Go for Training ---')
		for epo in range(train_epoch):
			for i, (v_d, v_p, label) in enumerate(training_generator):
				score = self.model(v_d.float().to(self.device), v_p.float().to(self.device))
				label = Variable(torch.from_numpy(np.array(label)).float())

				if self.binary:
					loss_fct = torch.nn.BCELoss()
					m = torch.nn.Sigmoid()
					n = torch.squeeze(m(score), 1)
					loss = loss_fct(n, label)
				else:
					loss_fct = torch.nn.MSELoss()
					n = torch.squeeze(score, 1)
					loss = loss_fct(n, label)
				loss_history.append(loss)

				opt.zero_grad()
				loss.backward()
				opt.step()

				if (i % 100 == 0):
					print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))

			with torch.set_grad_enabled(False):
				if self.binary:
					auc, auprc, f1, logits = self.test_(validation_generator, self.model)
					if auc > max_auc:
						model_max = copy.deepcopy(self.model)
						max_auc = auc   
					print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))
				else:
					mse, r2, p_val, CI, logits = self.test_(validation_generator, self.model)
					if mse < max_MSE:
						model_max = copy.deepcopy(self.model)
						max_MSE = mse
					print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) + ' with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
			
		if test is not None:
			print('--- Go for Testing ---')
			if self.binary:
				auc, auprc, f1, logits = self.test_(testing_generator, model_max)
				print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))				
			else:
				mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max)
				print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) + ' with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
		# load early stopped model
		self.model = model_max
		print('--- Training Finished ---')

	def predict(self, X_test):
		print('predicting...')
		v_d, v_p = X_test
		score = self.model(v_d.float().to(self.device), v_p.float().to(self.device))
		if self.binary:
			m = torch.nn.Sigmoid()
			logits = torch.squeeze(m(score)).detach().cpu().numpy()
			score = np.asarray([1 if i else 0 for i in (np.asarray(logits) >= 0.5)])
		return score

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)

	def load_pretrained(self, path):
		self.model.load_state_dict(torch.load(path))

