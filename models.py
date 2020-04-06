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
torch.manual_seed(2)    # reproducible torch:2 np:3
np.random.seed(3)
import copy
from prettytable import PrettyTable
import scikitplot as skplt

import os

if os.getcwd()[-7:] != 'Purpose':
	os.chdir('./DeepPurpose/')

from utils import *
from model_helper import Encoder_MultipleLayers, Embeddings    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class transformer(nn.Sequential):
	def __init__(self, encoding, **config):
		super(transformer, self).__init__()
		if encoding == 'drug':
			self.emb = Embeddings(config['input_dim_drug'], config['transformer_emb_size_drug'], 50, config['transformer_dropout_rate'])
			self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_drug'], 
													config['transformer_emb_size_drug'], 
													config['transformer_intermediate_size_drug'], 
													config['transformer_num_attention_heads_drug'],
													config['transformer_attention_probs_dropout'],
													config['transformer_hidden_dropout_rate'])
		elif encoding == 'protein':
			self.emb = Embeddings(config['input_dim_protein'], config['transformer_emb_size_target'], 545, config['transformer_dropout_rate'])
			self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_target'], 
													config['transformer_emb_size_target'], 
													config['transformer_intermediate_size_target'], 
													config['transformer_num_attention_heads_target'],
													config['transformer_attention_probs_dropout'],
													config['transformer_hidden_dropout_rate'])

	def forward(self, v):
		e = v[0].long().to(device)
		e_mask = v[1].long().to(device)
		ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
		ex_e_mask = (1.0 - ex_e_mask) * -10000.0

		emb = self.emb(e)
		encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
		return encoded_layers[:,0]


class CNN(nn.Sequential):
	def __init__(self, encoding, **config):
		super(CNN, self).__init__()
		if encoding == 'drug':
			in_ch = [63] + config['cnn_drug_filters']
			kernels = config['cnn_drug_kernels']
			layer_size = len(config['cnn_drug_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_d = self._get_conv_output((63, 100))
			#n_size_d = 1000
			self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

		if encoding == 'protein':
			in_ch = [26] + config['cnn_target_filters']
			kernels = config['cnn_target_kernels']
			layer_size = len(config['cnn_target_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_p = self._get_conv_output((26, 1000))

			self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input.double())
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def _forward_features(self, x):
		for l in self.conv:
			x = F.relu(l(x))
		x = F.adaptive_max_pool1d(x, output_size=1)
		return x

	def forward(self, v):
		v = self._forward_features(v.double())
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v


class CNN_RNN(nn.Sequential):
	def __init__(self, encoding, **config):
		super(CNN_RNN, self).__init__()
		if encoding == 'drug':
			in_ch = [63] + config['cnn_drug_filters']
			self.in_ch = in_ch[-1]
			kernels = config['cnn_drug_kernels']
			layer_size = len(config['cnn_drug_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_d = self._get_conv_output((63, 100)) # auto get the seq_len of CNN output

			if config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
				self.rnn = nn.LSTM(input_size = in_ch[-1], 
								hidden_size = config['rnn_drug_hid_dim'],
								num_layers = config['rnn_drug_n_layers'],
								batch_first = True,
								bidirectional = config['rnn_drug_bidirectional'])
			
			elif config['rnn_Use_GRU_LSTM_drug'] == 'GRU':
				self.rnn = nn.GRU(input_size = in_ch[-1], 
								hidden_size = config['rnn_drug_hid_dim'],
								num_layers = config['rnn_drug_n_layers'],
								batch_first = True,
								bidirectional = config['rnn_drug_bidirectional'])
			else:
				raise AttributeError('Please use LSTM or GRU.')
			self.rnn = self.rnn.double()
			self.fc1 = nn.Linear(config['rnn_drug_hid_dim'] * config['rnn_drug_n_layers'] * n_size_d, config['hidden_dim_drug'])

		if encoding == 'protein':
			in_ch = [26] + config['cnn_target_filters']
			self.in_ch = in_ch[-1]
			kernels = config['cnn_target_kernels']
			layer_size = len(config['cnn_target_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_p = self._get_conv_output((26, 1000))

			if config['rnn_Use_GRU_LSTM_target'] == 'LSTM':
				self.rnn = nn.LSTM(input_size = in_ch[-1], 
								hidden_size = config['rnn_target_hid_dim'],
								num_layers = config['rnn_target_n_layers'],
								batch_first = True,
								bidirectional = config['rnn_target_bidirectional'])

			elif config['rnn_Use_GRU_LSTM_target'] == 'GRU':
				self.rnn = nn.GRU(input_size = in_ch[-1], 
								hidden_size = config['rnn_target_hid_dim'],
								num_layers = config['rnn_target_n_layers'],
								batch_first = True,
								bidirectional = config['rnn_target_bidirectional'])
			else:
				raise AttributeError('Please use LSTM or GRU.')

			self.rnn = self.rnn.double()
			self.fc1 = nn.Linear(config['rnn_target_hid_dim'] * config['rnn_target_n_layers'] * n_size_p, config['hidden_dim_protein'])
		self.encoding = encoding
		self.config = config

	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input.double())
		n_size = output_feat.data.view(bs, self.in_ch, -1).size(2)
		return n_size

	def _forward_features(self, x):
		for l in self.conv:
			x = F.relu(l(x))
		return x

	def forward(self, v):
		for l in self.conv:
			v = F.relu(l(v.double()))
		batch_size = v.size(0)
		v = v.view(v.size(0), v.size(2), -1)

		if self.encoding == 'protein':
			if self.config['rnn_Use_GRU_LSTM_target'] == 'LSTM':
				direction = 2 if self.config['rnn_target_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size, self.config['rnn_target_hid_dim'])
				c0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size, self.config['rnn_target_hid_dim'])
				v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
			else:
				# GRU
				direction = 2 if self.config['rnn_target_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size, self.config['rnn_target_hid_dim'])
				v, hn = self.rnn(v.double(), h0.double())
		else:
			if self.config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
				direction = 2 if self.config['rnn_drug_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim'])
				c0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim'])
				v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
			else:
				# GRU
				direction = 2 if self.config['rnn_drug_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim'])
				v, hn = self.rnn(v.double(), h0.double())
		v = torch.flatten(v, 1)
		v = self.fc1(v.float())
		return v


class MLP(nn.Sequential):
	def __init__(self, input_dim, hidden_dim, hidden_dims):
		super(MLP, self).__init__()
		layer_size = len(hidden_dims) + 1
		dims = [input_dim] + hidden_dims + [hidden_dim]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v):
		# predict
		v = v.float().to(device)
		for i, l in enumerate(self.predictor):
			v = F.relu(l(v))
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

	def forward(self, feature):
		''' 
			batch_size == 1 
			feature: utils.smiles2mpnnfeature 
		'''
		fatoms, fbonds, agraph, bgraph, atoms_bonds = feature
		agraph = agraph.long()
		bgraph = bgraph.long()
		#print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape, atoms_bonds.shape)
		atoms_bonds = atoms_bonds.long()
		batch_size = atoms_bonds.shape[0]
		N_atoms, N_bonds = 0, 0 
		embeddings = []
		for i in range(batch_size):
			n_a = atoms_bonds[i,0].item()
			n_b = atoms_bonds[i,1].item()
			if (n_a == 0):
				embed = create_var(torch.zeros(1, self.mpnn_hidden_size))
				embeddings.append(embed)
				continue 
			sub_fatoms = fatoms[N_atoms:N_atoms+n_a,:].to(device)
			sub_fbonds = fbonds[N_bonds:N_bonds+n_b,:].to(device)
			sub_agraph = agraph[N_atoms:N_atoms+n_a,:].to(device)
			sub_bgraph = bgraph[N_bonds:N_bonds+n_b,:].to(device)
			embed = self.single_molecule_forward(sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
			embeddings.append(embed.to(device))
			N_atoms += n_a
			N_bonds += n_b
		try:
			embeddings = torch.cat(embeddings, 0)
		except:
			print(embeddings)
		return embeddings


	def single_molecule_forward(self, fatoms, fbonds, agraph, bgraph):
		'''
			fatoms: (x, 39)
			fbonds: (y, 50)
			agraph: (x, 6)
			bgraph: (y,6)
		'''
		fatoms = create_var(fatoms)
		fbonds = create_var(fbonds)
		agraph = create_var(agraph)
		bgraph = create_var(bgraph)

		binput = self.W_i(fbonds)
		message = F.relu(binput)
		#print("shapes", fbonds.shape, binput.shape, message.shape)
		for i in range(self.mpnn_depth - 1):
			nei_message = index_select_ND(message, 0, bgraph)
			nei_message = nei_message.sum(dim=1)
			nei_message = self.W_h(nei_message)
			message = F.relu(binput + nei_message)

		nei_message = index_select_ND(message, 0, agraph)
		nei_message = nei_message.sum(dim=1)
		ainput = torch.cat([fatoms, nei_message], dim=1)
		atom_hiddens = F.relu(self.W_o(ainput))
		return torch.mean(atom_hiddens, 0).view(1,-1).to(device)



class Classifier(nn.Sequential):
	def __init__(self, model_drug, model_protein, **config):
		super(Classifier, self).__init__()
		self.input_dim_drug = config['hidden_dim_drug']
		self.input_dim_protein = config['hidden_dim_protein']

		self.model_drug = model_drug
		self.model_protein = model_protein

		self.dropout = nn.Dropout(0.1)

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
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f    

def model_initialize(**config):
	model = DBTA(**config)
	return model

def model_pretrained(path_dir):
	config = load_dict(path_dir)
	model = DBTA(**config)
	model.load_pretrained(path_dir + '/model.pt')
	return model

def repurpose(X_repurpose, target, model, drug_names = None, target_name = None, result_folder = "./result/", convert_y = False, output_num_max = 10, verbose = True):
	# X_repurpose: a list of SMILES string
	# target: one target 
	fo = os.path.join(result_folder, "repurposing.txt")
	print_list = []
	with open(fo, 'w') as fout:
		print('repurposing...')
		df_data = data_process_repurpose_virtual_screening(X_repurpose, target, model.drug_encoding, model.target_encoding, 'repurposing')
		y_pred = model.predict(df_data)

		if convert_y:
			y_pred = convert_y_unit(np.array(y_pred), 'p', 'nM')

		print('---------------')
		if target_name is not None:
			if verbose:
				print('Drug Repurposing Result for '+target_name)
		if model.binary:
			table_header = ["Rank", "Drug Name", "Target Name", "Interaction", "Probability"]
		else:
			### regression 
			table_header = ["Rank", "Drug Name", "Target Name", "Binding Score"]
		table = PrettyTable(table_header)

		if drug_names is not None:
			f_d = max([len(o) for o in drug_names]) + 1
			for i in range(len(X_repurpose)):
				if model.binary:
					if y_pred[i] > 0.5:
						string_lst = [drug_names[i], target_name, "YES", "{0:.2f}".format(y_pred[i])]
						
					else:
						string_lst = [drug_names[i], target_name, "NO", "{0:.2f}".format(y_pred[i])]
				else:
					#### regression 
					#### Rank, Drug Name, Target Name, binding score 
					string_lst = [drug_names[i], target_name, "{0:.2f}".format(y_pred[i])]
					string = 'Drug ' + '{:<{f_d}}'.format(drug_names[i], f_d =f_d) + \
						' predicted to have binding affinity score ' + "{0:.2f}".format(y_pred[i])
					#print_list.append((string, y_pred[i]))
				print_list.append((string_lst, y_pred[i]))
		
		if convert_y:
			print_list.sort(key = lambda x:x[1])
		else:
			print_list.sort(key = lambda x:x[1], reverse = True)

		print_list = [i[0] for i in print_list]
		for idx, lst in enumerate(print_list):
			lst = [str(idx + 1)] + lst 
			table.add_row(lst)
		fout.write(table.get_string())
	if verbose:
		with open(fo, 'r') as fin:
			lines = fin.readlines()
			for idx, line in enumerate(lines):
				if idx < 13:
					print(line, end = '')
				else:
					print('checkout ' + fo + ' for the whole list')
					break
	return y_pred

def virtual_screening(X_repurpose, target, model, drug_names = None, target_names = None, result_folder = "./result/", convert_y = False, output_num_max = 10, verbose = True):
	# X_repurpose: a list of SMILES string
	# target: a list of targets
	fo = os.path.join(result_folder, "virtual_screening.txt")
	#if not model.binary:
	#	print_list = []
	print_list = []

	if model.binary:
		table_header = ["Rank", "Drug Name", "Target Name", "Interaction", "Probability"]
	else:
		### regression 
		table_header = ["Rank", "Drug Name", "Target Name", "Binding Score"]
	table = PrettyTable(table_header)

	with open(fo,'w') as fout:
		print('virtual screening...')
		df_data = data_process_repurpose_virtual_screening(X_repurpose, target, model.drug_encoding, model.target_encoding, 'virtual screening')
		y_pred = model.predict(df_data)

		if convert_y:
			y_pred = convert_y_unit(np.array(y_pred), 'p', 'nM')

		print('---------------')
		if drug_names is not None and target_names is not None:
			if verbose:
				print('Virtual Screening Result')
			f_d = max([len(o) for o in drug_names]) + 1
			f_p = max([len(o) for o in target_names]) + 1
			for i in range(target.shape[0]):
				if model.binary:
					if y_pred[i] > 0.5:
						string_lst = [drug_names[i], target_names[i], "YES", "{0:.2f}".format(y_pred[i])]						
						
					else:
						string_lst = [drug_names[i], target_names[i], "NO", "{0:.2f}".format(y_pred[i])]
						
				else:
					### regression 
					string_lst = [drug_names[i], target_names[i], "{0:.2f}".format(y_pred[i])]
					
				print_list.append((string_lst, y_pred[i]))
		print_list.sort(key = lambda x:x[1], reverse = True)
		print_list = [i[0] for i in print_list]
		for idx, lst in enumerate(print_list):
			lst = [str(idx+1)] + lst
			table.add_row(lst)
		fout.write(table.get_string())

	if verbose:
		with open(fo, 'r') as fin:
			lines = fin.readlines()
			for idx, line in enumerate(lines):
				if idx < 13:
					print(line, end = '')
				else:
					print('checkout ' + fo + ' for the whole list')
					break
		print()

	return y_pred

## x is a list, len(x)=batch_size, x[i] is tuple, len(x[0])=5  
def mpnn_feature_collate_func(x): 
	return [torch.cat([x[j][i] for j in range(len(x))], 0) for i in range(len(x[0]))]

def mpnn_collate_func(x):
	#print("len(x) is ", len(x)) ## batch_size 
	#print("len(x[0]) is ", len(x[0])) ## 3--- data_process_loader.__getitem__ 
	mpnn_feature = [i[0] for i in x]
	#print("len(mpnn_feature)", len(mpnn_feature), "len(mpnn_feature[0])", len(mpnn_feature[0]))
	mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
	from torch.utils.data.dataloader import default_collate
	x_remain = [[i[1], i[2]] for i in x]
	x_remain_collated = default_collate(x_remain)
	return [mpnn_feature] + x_remain_collated
## used in dataloader 


class DBTA:
	def __init__(self, **config):
		drug_encoding = config['drug_encoding']
		target_encoding = config['target_encoding']

		if drug_encoding == 'Morgan' or drug_encoding=='Pubchem' or drug_encoding=='Daylight' or drug_encoding=='rdkit_2d_normalized':
			# Future TODO: support multiple encoding scheme for static input 
			self.model_drug = MLP(config['input_dim_drug'], config['hidden_dim_drug'], config['mlp_hidden_dims_drug'])
		elif drug_encoding == 'CNN':
			self.model_drug = CNN('drug', **config)
		elif drug_encoding == 'CNN_RNN':
			self.model_drug = CNN_RNN('drug', **config)
		elif drug_encoding == 'Transformer':
			self.model_drug = transformer('drug', **config)
		elif drug_encoding == 'MPNN':
			self.model_drug = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])
		else:
			raise AttributeError('Please use one of the available encoding method.')

		if target_encoding == 'AAC' or target_encoding == 'PseudoAAC' or target_encoding == 'Conjoint_triad' or target_encoding == 'Quasi-seq':
			self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'], config['mlp_hidden_dims_target'])
		elif target_encoding == 'CNN':
			self.model_protein = CNN('protein', **config)
		elif target_encoding == 'CNN_RNN':
			self.model_protein = CNN_RNN('protein', **config)
		elif target_encoding == 'Transformer':
			self.model_protein = transformer('protein', **config)
		else:
			raise AttributeError('Please use one of the available encoding method.')

		self.model = Classifier(self.model_drug, self.model_protein, **config)
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.drug_encoding = drug_encoding
		self.target_encoding = target_encoding
		self.result_folder = config['result_folder']
		self.binary = False


	def test_(self, data_generator, model, repurposing_mode = False, test = False):
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_d, v_p, label) in enumerate(data_generator):
			if self.drug_encoding == "MPNN":
				score = self.model(v_d, v_p.float().to(self.device))
			else:
				score = self.model(v_d.float().to(self.device), v_p.float().to(self.device))
			#score = model(v_d, v_p)
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
			if repurposing_mode:
				return y_pred
			## ROC-AUC curve
			if test:
				roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
				plt.figure(0)
				roc_curve(y_pred, y_label, roc_auc_file)
				plt.figure(1)
				pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
				prauc_curve(y_pred, y_label, pr_auc_file)

			return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred
		else:
			if repurposing_mode:
				return y_pred
			return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred

	def train(self, train, val, test = None, verbose = True):
		if len(train.Label.unique()) == 2:
			self.binary = True
			self.config['binary'] = True

		lr = self.config['LR']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		loss_history = []

		self.model = self.model.to(self.device)

		# support multiple GPUs
		if torch.cuda.device_count() >= 1:
			print("Let's use " + str(torch.cuda.device_count()) + " GPU/s!")
			self.model = nn.DataParallel(self.model, dim = 0)

		# Future TODO: support multiple optimizers with parameters
		opt = torch.optim.Adam(self.model.parameters(), lr = lr)

		print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': 0,
	    		'drop_last': False}

		if (self.drug_encoding == "MPNN"):
			params['collate_fn'] = mpnn_collate_func

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

		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		if self.binary:
			valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
		else:
			valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
		table = PrettyTable(valid_metric_header)
		float2str = lambda x:'%0.4f'%x

		print('--- Go for Training ---')
		for epo in range(train_epoch):
			for i, (v_d, v_p, label) in enumerate(training_generator):
				if (self.drug_encoding == "MPNN"):
					score = self.model(v_d, v_p.float().to(self.device))
				else:
					score = self.model(v_d.float().to(self.device), v_p.float().to(self.device))
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
						print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))

			with torch.set_grad_enabled(False):
				if self.binary:  
					## binary: ROC-AUC, PR-AUC, F1  
					auc, auprc, f1, logits = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
					valid_metric_record.append(lst)
					if auc > max_auc:
						model_max = copy.deepcopy(self.model)
						max_auc = auc   
					print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))
				else:  
					### regression: MSE, Pearson Correlation, with p-value, Concordance Index  
					mse, r2, p_val, CI, logits = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
					valid_metric_record.append(lst)
					if mse < max_MSE:
						model_max = copy.deepcopy(self.model)
						max_MSE = mse
					print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse) + ' , Pearson Correlation: '\
						 + str(r2) + ' with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
			table.add_row(lst)
		prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(table.get_string())



		if test is not None:
			print('--- Go for Testing ---')
			if self.binary:
				auc, auprc, f1, logits = self.test_(testing_generator, model_max, test = True)
				test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
				test_table.add_row(list(map(float2str, [auc, auprc, f1])))
				print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))				
			else:
				mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max)
				test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
				test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
				print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) + ' with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
		# load early stopped model
		self.model = model_max


		######### learning record ###########

		### 1. test results
		prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(test_table.get_string())

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

		print('--- Training Finished ---')


	def predict(self, df_data):
		print('predicting...')
		info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
		
		params = {'batch_size': self.config['batch_size'],
				'shuffle': False,
				'num_workers': 0,
				'drop_last': False,
				'sampler':SequentialSampler(info)}

		if (self.drug_encoding == "MPNN"):
			params['collate_fn'] = mpnn_collate_func


		generator = data.DataLoader(info, **params)

		score = self.test_(generator, self.model, repurposing_mode = True)
		return score

	def save_model(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(self.model.state_dict(), path_dir + '/model.pt')
		save_dict(path_dir, self.config)

	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)
		self.model.load_state_dict(torch.load(path))
		self.binary = self.config['binary']


