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
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
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

	### parameter v (tuple of length 2) is from utils.drug2emb_encoder 
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
			direction = 2 if config['rnn_drug_bidirectional'] else 1
			self.rnn = self.rnn.double()
			self.fc1 = nn.Linear(config['rnn_drug_hid_dim'] * direction * n_size_d, config['hidden_dim_drug'])

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
			direction = 2 if config['rnn_target_bidirectional'] else 1
			self.rnn = self.rnn.double()
			self.fc1 = nn.Linear(config['rnn_target_hid_dim'] * direction * n_size_p, config['hidden_dim_protein'])
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
				h0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size, self.config['rnn_target_hid_dim']).to(device)
				c0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size, self.config['rnn_target_hid_dim']).to(device)
				v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
			else:
				# GRU
				direction = 2 if self.config['rnn_target_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size, self.config['rnn_target_hid_dim']).to(device)
				v, hn = self.rnn(v.double(), h0.double())
		else:
			if self.config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
				direction = 2 if self.config['rnn_drug_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim']).to(device)
				c0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim']).to(device)
				v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
			else:
				# GRU
				direction = 2 if self.config['rnn_drug_bidirectional'] else 1
				h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim']).to(device)
				v, hn = self.rnn(v.double(), h0.double())
		v = torch.flatten(v, 1)
		v = self.fc1(v.float())
		return v


class MLP(nn.Sequential):
	def __init__(self, input_dim, output_dim, hidden_dims_lst):
		'''
			input_dim (int)
			output_dim (int)
			hidden_dims_lst (list, each element is a integer, indicating the hidden size)

		'''
		super(MLP, self).__init__()
		layer_size = len(hidden_dims_lst) + 1
		dims = [input_dim] + hidden_dims_lst + [output_dim]

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
		from DeepPurpose.chemutils import ATOM_FDIM, BOND_FDIM

		self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
		self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
		self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)



	## utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward
	def forward(self, feature):
		'''
			fatoms: (x, 39)
			fbonds: (y, 50)
			agraph: (x, 6)
			bgraph: (y, 6)
		'''
		fatoms, fbonds, agraph, bgraph, N_atoms_bond = feature
		N_atoms_scope = []
		##### tensor feature -> matrix feature
		N_a, N_b = 0, 0 
		fatoms_lst, fbonds_lst, agraph_lst, bgraph_lst = [],[],[],[]
		for i in range(N_atoms_bond.shape[0]):
			atom_num = int(N_atoms_bond[i][0].item()) 
			bond_num = int(N_atoms_bond[i][1].item()) 

			fatoms_lst.append(fatoms[i,:atom_num,:])
			fbonds_lst.append(fbonds[i,:bond_num,:])
			agraph_lst.append(agraph[i,:atom_num,:] + N_a)
			bgraph_lst.append(bgraph[i,:bond_num,:] + N_b)

			N_atoms_scope.append((N_a, atom_num))
			N_a += atom_num 
			N_b += bond_num 


		fatoms = torch.cat(fatoms_lst, 0)
		fbonds = torch.cat(fbonds_lst, 0)
		agraph = torch.cat(agraph_lst, 0)
		bgraph = torch.cat(bgraph_lst, 0)
		##### tensor feature -> matrix feature


		agraph = agraph.long()
		bgraph = bgraph.long()	

		fatoms = create_var(fatoms).to(device)
		fbonds = create_var(fbonds).to(device)
		agraph = create_var(agraph).to(device)
		bgraph = create_var(bgraph).to(device)

		binput = self.W_i(fbonds) #### (y, d1)
		message = F.relu(binput)  #### (y, d1)		

		for i in range(self.mpnn_depth - 1):
			nei_message = index_select_ND(message, 0, bgraph)
			nei_message = nei_message.sum(dim=1)
			nei_message = self.W_h(nei_message)
			message = F.relu(binput + nei_message) ### (y,d1) 

		nei_message = index_select_ND(message, 0, agraph)
		nei_message = nei_message.sum(dim=1)
		ainput = torch.cat([fatoms, nei_message], dim=1)
		atom_hiddens = F.relu(self.W_o(ainput))
		output = [torch.mean(atom_hiddens.narrow(0, sts,leng), 0) for sts,leng in N_atoms_scope]
		output = torch.stack(output, 0)
		return output 


class DGL_GCN(nn.Module):
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/gcn_predictor.py#L16
	def __init__(self, in_feats, hidden_feats=None, activation=None, predictor_dim=None):
		super(DGL_GCN, self).__init__()
		from dgllife.model.gnn.gcn import GCN
		from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

		self.gnn = GCN(in_feats=in_feats,
						hidden_feats=hidden_feats,
						activation=activation
						)
		gnn_out_feats = self.gnn.hidden_feats[-1]
		self.readout = WeightedSumAndMax(gnn_out_feats)
		self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2, predictor_dim)

	def forward(self, bg):
		bg = bg.to(device)
		feats = bg.ndata.pop('h') 
		node_feats = self.gnn(bg, feats)
		graph_feats = self.readout(bg, node_feats)
		return self.transform(graph_feats)

class DGL_NeuralFP(nn.Module):
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/gat_predictor.py
	def __init__(self, in_feats, hidden_feats=None, max_degree = None, activation=None, predictor_hidden_size = None, predictor_activation = None, predictor_dim=None):
		super(DGL_NeuralFP, self).__init__()
		from dgllife.model.gnn.nf import NFGNN
		from dgllife.model.readout.sum_and_max import SumAndMax

		self.gnn = NFGNN(in_feats=in_feats,
						hidden_feats=hidden_feats,
						max_degree=max_degree,
						activation=activation
						)
		gnn_out_feats = self.gnn.gnn_layers[-1].out_feats
		self.node_to_graph = nn.Linear(gnn_out_feats, predictor_hidden_size)
		self.predictor_activation = predictor_activation

		self.readout = SumAndMax()
		self.transform = nn.Linear(predictor_hidden_size * 2, predictor_dim)

	def forward(self, bg):
		bg = bg.to(device)        
		feats = bg.ndata.pop('h') 
		node_feats = self.gnn(bg, feats)
		node_feats = self.node_to_graph(node_feats)
		graph_feats = self.readout(bg, node_feats)
		graph_feats = self.predictor_activation(graph_feats)
		return self.transform(graph_feats)


class DGL_GIN_AttrMasking(nn.Module):
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
	def __init__(self, predictor_dim=None):
		super(DGL_GIN_AttrMasking, self).__init__()
		from dgllife.model import load_pretrained
		from dgl.nn.pytorch.glob import AvgPooling

		## this is fixed hyperparameters as it is a pretrained model
		self.gnn = load_pretrained('gin_supervised_masking')

		self.readout = AvgPooling()
		self.transform = nn.Linear(300, predictor_dim)

	def forward(self, bg):
		bg = bg.to(device)                
		node_feats = [
			bg.ndata.pop('atomic_number'),
			bg.ndata.pop('chirality_type')
		]
		edge_feats = [
			bg.edata.pop('bond_type'),
			bg.edata.pop('bond_direction_type')
		]

		node_feats = self.gnn(bg, node_feats, edge_feats)
		graph_feats = self.readout(bg, node_feats)
		return self.transform(graph_feats)

class DGL_GIN_ContextPred(nn.Module):
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
	def __init__(self, predictor_dim=None):
		super(DGL_GIN_ContextPred, self).__init__()
		from dgllife.model import load_pretrained
		from dgl.nn.pytorch.glob import AvgPooling

		## this is fixed hyperparameters as it is a pretrained model
		self.gnn = load_pretrained('gin_supervised_contextpred')

		self.readout = AvgPooling()
		self.transform = nn.Linear(300, predictor_dim)

	def forward(self, bg):
		bg = bg.to(device)                
		node_feats = [
			bg.ndata.pop('atomic_number'),
			bg.ndata.pop('chirality_type')
		]
		edge_feats = [
			bg.edata.pop('bond_type'),
			bg.edata.pop('bond_direction_type')
		]

		node_feats = self.gnn(bg, node_feats, edge_feats)
		graph_feats = self.readout(bg, node_feats)
		return self.transform(graph_feats)


class DGL_AttentiveFP(nn.Module):
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/attentivefp_predictor.py#L17
	def __init__(self, node_feat_size, edge_feat_size, num_layers = 2, num_timesteps = 2, graph_feat_size = 200, predictor_dim=None):
		super(DGL_AttentiveFP, self).__init__()
		from dgllife.model.gnn import AttentiveFPGNN
		from dgllife.model.readout import AttentiveFPReadout

		self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size)

		self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps)

		self.transform = nn.Linear(graph_feat_size, predictor_dim)

	def forward(self, bg):
		bg = bg.to(device)                
		node_feats = bg.ndata.pop('h')
		edge_feats = bg.edata.pop('e')

		node_feats = self.gnn(bg, node_feats, edge_feats)
		graph_feats = self.readout(bg, node_feats, False)
		return self.transform(graph_feats)
