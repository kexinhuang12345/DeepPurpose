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
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

torch.manual_seed(2)    # reproducible torch:2 np:3
np.random.seed(3)
import copy

from utils import data_process_loader

class LR(nn.Sequential):
	def __init__(self, **config):
		super(LR, self).__init__()
		self.input_dim = config['input_dim']
		# predictor: LR
		self.predictor = nn.Sequential(
			nn.Linear(self.input_dim, 1),
		)

	def forward(self, v_D):
		# predict
		score = self.predictor(v_D)
		return score
    
    
class MLP(nn.Sequential):
	def __init__(self, input_dim, hidden_dim):
		super(MLP, self).__init__()

		self.predictor = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.Linear(1024, 512),
			nn.Linear(512, 256),
			nn.Linear(256, hidden_dim)            
		)

	def forward(self, v_D):
		# predict
		hid = self.predictor(v_D)
		return hid    


class Classifier(nn.Sequential):
	def __init__(self, **config, model_drug, model_protein):
		super(Classifier, self).__init__()
		self.input_dim = config['hidden_dim_drug']
		self.input_dim_protein = config['hidden_dim_protein']

		self.model_drug = model_drug
		self.model_protein = model_protein

		self.predictor = nn.Sequential(
			nn.Linear(self.input_dim_drug + self.input_dim_protein, 1024),
			nn.Linear(1024, 512),
			nn.Linear(512, 256),
			nn.Linear(256, 1)            
		)

	def forward(self, v_D, v_P):
		# each encoding
		v_D = self.model_drug(v_D)
		v_P = self.model_protein(v_P)
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 0)
		score = self.predictor(v_f)
		return score    


def model_initialize(drug_encoding, target_encoding, **config):
	model = DBTA(drug_encoding, target_encoding, **config)
	return model

def model_pretrained(path, drug_encoding, target_encoding, **config):
	model = DBTA(drug_encoding, target_encoding, **config)
	model.model.load_state_dict(torch.load(path))
	return model

def repurpose(X_repurpose, model_Kd = None, model_IC50 = None):
	if model_Kd is None & model_IC50 is None:
		raise AttributeError('Need at least a model, either Kd or IC50.')
	elif model_Kd not None & model_IC50 is None:
		print('Repurposing to generate Kd values...')
		Kd = model_Kd.predict(X_repurpose)
		return Kd
	elif model_IC50 not None & model_Kd is None:
		print('Repurposing to generate IC50 values...')
		IC50 = model_IC50.predict(X_repurpose)
		return IC50
	else:
		print('Repurposing to generate Kd & IC50 values...')
		Kd = model_Kd.predict(X_repurpose)
		IC50 = model_IC50.predict(X_repurpose)
		return Kd, IC50

class DBTA:
	def __init__(self, drug_encoding, target_encoding, **config):
		if drug_encoding == 'ECFP4':
			#TODO: support multiple encoding scheme for static input 
			self.model_drug = MLP(config['input_dim_drug'], config['hidden_dim_drug'])
		else:
			raise AttributeError('Please use one of the available encoding method.')

		if target_encoding == 'AAC':
			self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'])

		self.model = Classifier(**config, self.model_drug, self.model_protein)
		self.config = config

	def train(train, val, test = None):
		lr = self.config['LR']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		loss_history = []

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(device)

		if torch.cuda.device_count() > 1:
	        print("Let's use", torch.cuda.device_count(), "GPUs!")
	        self.model = nn.DataParallel(self.model, dim = 0)

		# TODO: support multiple optimizers with parameters
		opt = torch.optim.Adam(model.parameters(), lr = lr)

		print('--- Data Preparation ---')
    
	    params = {'batch_size': BATCH_SIZE,
	              'shuffle': True,
	              'num_workers': 0, 
	              'drop_last': True}

	    training_generator = data.DataLoader(data_process_loader(train.index.values, train.Label.values, train, config))
	    validation_generator = data.DataLoader(data_process_loader(val.index.values, val.Label.values, val, config))
	    testing_generator = data.DataLoader(data_process_loader(test.index.values, test.Label.values, test, config))

	    # early stopping
	    max_MSE = 10000
	    model_max = copy.deepcopy(self.model)

	    print('--- Go for Training ---')
	    for epo in range(train_epoch):
	    
	        for i, (v_d, v_p, label) in enumerate(training_generator):
	            score = self.model(v_d.float().to(device), v_p.float().to(device))
	            label = Variable(torch.from_numpy(np.array(label)).float())
	            
	            loss_fct = torch.nn.MSELoss() 
	            n = torch.squeeze(score)
	            
	            loss = loss_fct(n, label)
	            loss_history.append(loss)
	            
	            opt.zero_grad()
	            loss.backward()
	            opt.step()
	            
	            if (i % 100 == 0):
	                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))
	            
	        # every epoch validation
	        with torch.set_grad_enabled(False):
	            mse, r2, p_val, CI, logits = test(validation_generator, self.model)
	            if mse < max_mse:
	                model_max = copy.deepcopy(self.model)
	                max_mse = mse  
	            print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse) + ' , pearson correlation: ' + str(r2) + 'with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
	    
	    if test is not None:
	    	print('--- Go for Testing ---')
    		mse, r2, p_val, CI, logits = test(testing_generator, model_max)
    		print('Testing MSE: ' + str(mse) + ' , pearson correlation: ' + str(r2) + 'with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
    
    print('--- Training Finished ---')

	def test(data_generator, model):
	    y_pred = []
	    y_label = []
	    model.eval()
	    for i, (v_d, v_p, label) in enumerate(data_generator):
	        score = model(v_d.float().to(device), v_p.float().to(device))
	        logits = torch.squeeze(score).detach().cpu().numpy()
	        label_ids = label.to('cpu').numpy()
	        y_label = y_label + label_ids.flatten().tolist()
	        y_pred = y_pred + logits.flatten().tolist()
	    return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, outputs), y_pred


	def predict(X_test, Y_test = None):













