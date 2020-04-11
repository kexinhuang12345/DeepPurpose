import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, calcPubChemFingerAll, CalculateConjointTriad, GetQuasiSequenceOrder
import torch
from torch.utils import data
from torch.autograd import Variable
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from chemutils import get_mol, atom_features, bond_features, MAX_NB, ATOM_FDIM, BOND_FDIM
from subword_nmt.apply_bpe import BPE
import codecs
import pickle

import os
if os.getcwd()[-7:] != 'Purpose':
	os.chdir('./DeepPurpose/')
# ESPF encoding
vocab_path = './ESPF/drug_codes_chembl_freq_1500.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl_freq_1500.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

vocab_path = './ESPF/protein_codes_uniprot_2000.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
#sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot_2000.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

from chemutils import get_mol, atom_features, bond_features, MAX_NB

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def roc_curve(y_pred, y_label, figure_file, method_name):
	'''
		y_pred is a list of length n.  (0,1)
		y_label is a list of same length. 0/1
		https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
	'''
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve, auc
	from sklearn.metrics import roc_auc_score
	y_label = np.array(y_label)
	y_pred = np.array(y_pred)	
	fpr = dict()
	tpr = dict() 
	roc_auc = dict()
	fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
	roc_auc[0] = auc(fpr[0], tpr[0])
	lw = 2
	plt.plot(fpr[0], tpr[0],
         lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fontsize = 14
	plt.xlabel('False Positive Rate', fontsize = fontsize)
	plt.ylabel('True Positive Rate', fontsize = fontsize)
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	plt.savefig(figure_file)
	return 

def prauc_curve(y_pred, y_label, figure_file, method_name):
	'''
		y_pred is a list of length n.  (0,1)
		y_label is a list of same length. 0/1
		reference: 
			https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
	'''	
	import matplotlib.pyplot as plt
	from sklearn.metrics import precision_recall_curve, average_precision_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import auc
	lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
#	plt.plot([0,1], [no_skill, no_skill], linestyle='--')
	plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.2f)' % average_precision_score(y_label, y_pred))
	fontsize = 14
	plt.xlabel('Recall', fontsize = fontsize)
	plt.ylabel('Precision', fontsize = fontsize)
	plt.title('Precision Recall Curve')
	plt.legend()
	plt.savefig(figure_file)
	return 


def length_func(list_or_tensor):
	if type(list_or_tensor)==list:
		return len(list_or_tensor)
	return list_or_tensor.shape[0]

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def smiles2morgan(s, radius = 2, nBits = 1024):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 1 features')
        features = np.ones((nBits, ))
    return features

def smiles2rdkit2d(s):    
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(s)[1:]
    except:
        print('descriptastorus not found this smiles: ' + s + ' convert to all 1 features')
        features = np.ones((200, ))
    return np.array(features)

def smiles2daylight(s):
	try:
		NumFinger = 2048
		mol = Chem.MolFromSmiles(s)
		bv = FingerprintMols.FingerprintMol(mol)
		temp = tuple(bv.GetOnBits())
		features = np.zeros((NumFinger, ))
		features[np.array(temp)] = 1
	except:
		print('rdkit not found this smiles: ' + s + ' convert to all 1 features')
		features = np.ones((2048, ))
	return np.array(features)


def smiles2mpnnfeature(smiles):
	## mpn.py::tensorize  
	'''
		data-flow:   
			data_process(): apply(smiles2mpnnfeature)
			DBTA: train(): data.DataLoader(data_process_loader())
			mpnn_collate_func()
	'''
	try: 
		padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
		fatoms, fbonds = [], [padding] 
		in_bonds,all_bonds = [], [(-1,-1)] 
		mol = get_mol(smiles)
		n_atoms = mol.GetNumAtoms()
		for atom in mol.GetAtoms():
			fatoms.append( atom_features(atom))
			in_bonds.append([])

		for bond in mol.GetBonds():
			a1 = bond.GetBeginAtom()
			a2 = bond.GetEndAtom()
			x = a1.GetIdx() 
			y = a2.GetIdx()

			b = len(all_bonds)
			all_bonds.append((x,y))
			fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
			in_bonds[y].append(b)

			b = len(all_bonds)
			all_bonds.append((y,x))
			fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
			in_bonds[x].append(b)

		total_bonds = len(all_bonds)
		fatoms = torch.stack(fatoms, 0) 
		fbonds = torch.stack(fbonds, 0) 
		agraph = torch.zeros(n_atoms,MAX_NB).long()
		bgraph = torch.zeros(total_bonds,MAX_NB).long()
		for a in range(n_atoms):
			for i,b in enumerate(in_bonds[a]):
				agraph[a,i] = b

		for b1 in range(1, total_bonds):
			x,y = all_bonds[b1]
			for i,b2 in enumerate(in_bonds[x]):
				if all_bonds[b2][0] != y:
					bgraph[b1,i] = b2
	except: 
		print('Molecules not found and change to zero vectors..')
		fatoms = torch.zeros(0,39)
		fbonds = torch.zeros(0,50)
		agraph = torch.zeros(0,6)
		bgraph = torch.zeros(0,6)
	#fatoms, fbonds, agraph, bgraph = [], [], [], [] 
	#print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape)
	Natom, Nbond = fatoms.shape[0], fbonds.shape[0]
	shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
	return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()]



# random_fold
def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

# cold protein
def create_fold_setting_cold_protein(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['Target Sequence'].isin(gene_drop)]

    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test

# cold drug
def create_fold_setting_cold_drug(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['SMILES'].isin(drug_drop)]

    train_val = df[~df['SMILES'].isin(drug_drop)]
    
    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]
    
    return train, val, test

#TODO: add one target, drug folding

def data_process(X_drug, X_target, y=None, drug_encoding=None, target_encoding=None, split_method = 'random', frac = [0.7, 0.1, 0.2], random_seed = 1, sample_frac = 1):
	if split_method == 'repurposing_VS':
		y = [-1]*len(X_drug) # create temp y for compatitibility
	if isinstance(X_target, str):
		X_target = [X_target]
	if len(X_target) == 1:
		# one target high throughput screening setting
		X_target = np.tile(X_target, (length_func(X_drug), ))

	df_data = pd.DataFrame(zip(X_drug, X_target, y))
	df_data.rename(columns={0:'SMILES',
							1: 'Target Sequence',
							2: 'Label'}, 
							inplace=True)

	print('in total: ' + str(len(df_data)) + ' drug-target pairs')
    
	if sample_frac != 1:
		df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
		print('after subsample: ' + str(len(df_data)) + ' drug-target pairs')        
        
	print('encoding drug...')
	print('unique drugs: ' + str(len(df_data['SMILES'].unique())))

	if drug_encoding == 'Morgan':
		unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2morgan)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
	elif drug_encoding == 'Pubchem':
		unique = pd.Series(df_data['SMILES'].unique()).apply(calcPubChemFingerAll)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
	elif drug_encoding == 'Daylight':
		unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2daylight)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
	elif drug_encoding == 'rdkit_2d_normalized':
		try:
			unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2rdkit2d)
			unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
			df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
		except:
			raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus.")
	elif drug_encoding == 'CNN':
		unique = pd.Series(df_data['SMILES'].unique()).apply(trans_drug)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
	elif drug_encoding == 'CNN_RNN':
		unique = pd.Series(df_data['SMILES'].unique()).apply(trans_drug)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
	elif drug_encoding == 'Transformer':
		unique = pd.Series(df_data['SMILES'].unique()).apply(drug2emb_encoder)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
	elif drug_encoding == 'MPNN':
		#print(pd.Series(df_data['SMILES'].unique()))
		unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2mpnnfeature)
		unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
		df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]	
		#raise NotImplementedError
	else:
		raise AttributeError("Please use the correct drug encoding available!")

	print('drug encoding finished...')
	print('encoding protein...')
	print('unique target sequence: ' + str(len(df_data['Target Sequence'].unique())))

	if target_encoding == 'AAC':
		print('-- Encoding AAC takes time. Time Reference: 24s for ~100 sequences in a CPU. Calculate your time by the unique target sequence #, instead of the entire dataset.')
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(CalculateAADipeptideComposition)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
	elif target_encoding == 'PseudoAAC':
		print('-- Encoding PseudoAAC takes time. Time Reference: 462s for ~100 sequences in a CPU. Calculate your time by the unique target sequence #, instead of the entire dataset.')
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(_GetPseudoAAC)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
	elif target_encoding == 'Conjoint_triad':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(CalculateConjointTriad)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
	elif target_encoding == 'Quasi-seq':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(GetQuasiSequenceOrder)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
	elif target_encoding == 'CNN':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
	elif target_encoding == 'CNN_RNN':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
	elif target_encoding == 'Transformer':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(protein2emb_encoder)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
	else:
		raise AttributeError("Please use the correct protein encoding available!")

	print('protein encoding finished...')
	if split_method == 'repurposing_VS':
		pass
	else:
		print('splitting dataset...')

	if split_method == 'random': 
		train, val, test = create_fold(df_data, random_seed, frac)
	elif split_method == 'cold_drug':
		train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
	elif split_method == 'HTS':
		train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
		val = pd.concat([val[val.Label == 1].drop_duplicates(subset = 'SMILES'), val[val.Label == 0]])
		test = pd.concat([test[test.Label == 1].drop_duplicates(subset = 'SMILES'), test[test.Label == 0]])        
	elif split_method == 'cold_protein':
		train, val, test = create_fold_setting_cold_protein(df_data, random_seed, frac)
	elif split_method == 'repurposing_VS':
		train = df_data
		val = df_data
		test = df_data
	else:
		raise AttributeError("Please select one of the three split method: random, cold_drug, cold_target!")

	print('Done.')
	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def data_process_repurpose_virtual_screening(X_repurpose, target, drug_encoding, target_encoding, mode):
	if mode == 'repurposing':
		target = np.tile(target, (len(X_repurpose), ))
	elif mode == 'virtual screening':
		target = target
	else:
		raise AttributeError("Please select repurposing or virtual screening!")

	df, _, _ = data_process(X_repurpose, target, drug_encoding = drug_encoding, 
								target_encoding = target_encoding, 
								split_method='repurposing_VS')

	return df

class data_process_loader(data.Dataset):

	def __init__(self, list_IDs, labels, df, **config):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.df = df
		self.config = config

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs[index]
		v_d = self.df.iloc[index]['drug_encoding']        
		if self.config['drug_encoding'] == 'CNN' or self.config['drug_encoding'] == 'CNN_RNN':
			v_d = drug_2_embed(v_d)
		v_p = self.df.iloc[index]['target_encoding']
		if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
			v_p = protein_2_embed(v_p)
		y = self.labels[index]
		return v_d, v_p, y


def generate_config(drug_encoding, target_encoding, 
					result_folder = "./result/",
					input_dim_drug = 1024, 
					input_dim_protein = 8420,
					hidden_dim_drug = 256, 
					hidden_dim_protein = 256,
					cls_hidden_dims = [1024, 1024, 512],
					mlp_hidden_dims_drug = [1024, 256, 64],
					mlp_hidden_dims_target = [1024, 256, 64],
					batch_size = 256,
					train_epoch = 10,
					test_every_X_epoch = 20,
					LR = 1e-4,
					transformer_emb_size_drug = 128,
					transformer_intermediate_size_drug = 512,
					transformer_num_attention_heads_drug = 8,
					transformer_n_layer_drug = 8,
					transformer_emb_size_target = 128,
					transformer_intermediate_size_target = 512,
					transformer_num_attention_heads_target = 8,
					transformer_n_layer_target = 4,
					transformer_dropout_rate = 0.1,
					transformer_attention_probs_dropout = 0.1,
					transformer_hidden_dropout_rate = 0.1,
					mpnn_hidden_size = 50,
					mpnn_depth = 3,
					cnn_drug_filters = [32,64,96],
					cnn_drug_kernels = [4,6,8],
					cnn_target_filters = [32,64,96],
					cnn_target_kernels = [4,8,12],
					rnn_Use_GRU_LSTM_drug = 'GRU',
					rnn_drug_hid_dim = 64,
					rnn_drug_n_layers = 2,
					rnn_drug_bidirectional = True,
					rnn_Use_GRU_LSTM_target = 'GRU',
					rnn_target_hid_dim = 64,
					rnn_target_n_layers = 2,
					rnn_target_bidirectional = True
					):

	base_config = {'input_dim_drug': input_dim_drug,
					'input_dim_protein': input_dim_protein,
					'hidden_dim_drug': hidden_dim_drug, # hidden dim of drug
					'hidden_dim_protein': hidden_dim_protein, # hidden dim of protein
					'cls_hidden_dims' : cls_hidden_dims, # decoder classifier dim 1
					'batch_size': batch_size,
					'train_epoch': train_epoch,
					'test_every_X_epoch': test_every_X_epoch, 
					'LR': LR,
					'drug_encoding': drug_encoding,
					'target_encoding': target_encoding, 
					'result_folder': result_folder,
					'binary': False
	}
	if not os.path.exists(base_config['result_folder']):
		os.makedirs(base_config['result_folder'])
	
	if drug_encoding == 'Morgan':
		base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug # MLP classifier dim 1				
	elif drug_encoding == 'Pubchem':
		base_config['input_dim_drug'] = 881
		base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug # MLP classifier dim 1				
	elif drug_encoding == 'Daylight':
		base_config['input_dim_drug'] = 2048
		base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug # MLP classifier dim 1						
	elif drug_encoding == 'rdkit_2d_normalized':
		base_config['input_dim_drug'] = 200
		base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug # MLP classifier dim 1				
	elif drug_encoding == 'CNN':
		base_config['cnn_drug_filters'] = cnn_drug_filters
		base_config['cnn_drug_kernels'] = cnn_drug_kernels
	elif drug_encoding == 'CNN_RNN':
		base_config['rnn_Use_GRU_LSTM_drug'] = rnn_Use_GRU_LSTM_drug
		base_config['rnn_drug_hid_dim'] = rnn_drug_hid_dim
		base_config['rnn_drug_n_layers'] = rnn_drug_n_layers
		base_config['rnn_drug_bidirectional'] = rnn_drug_bidirectional 
		base_config['cnn_drug_filters'] = cnn_drug_filters
		base_config['cnn_drug_kernels'] = cnn_drug_kernels
	elif drug_encoding == 'Transformer':
		base_config['input_dim_drug'] = 2586
		base_config['transformer_emb_size_drug'] = transformer_emb_size_drug
		base_config['transformer_num_attention_heads_drug'] = transformer_num_attention_heads_drug
		base_config['transformer_intermediate_size_drug'] = transformer_intermediate_size_drug
		base_config['transformer_n_layer_drug'] = transformer_n_layer_drug
		base_config['transformer_dropout_rate'] = transformer_dropout_rate
		base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
		base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
		base_config['hidden_dim_drug'] = transformer_emb_size_drug
	elif drug_encoding == 'MPNN':
		base_config['hidden_dim_drug'] = hidden_dim_drug
		base_config['batch_size'] = batch_size 
		base_config['mpnn_hidden_size'] = mpnn_hidden_size
		base_config['mpnn_depth'] = mpnn_depth
		#raise NotImplementedError
	else:
		raise AttributeError("Please use the correct drug encoding available!")

	if target_encoding == 'AAC':
		base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	elif target_encoding == 'PseudoAAC':
		base_config['input_dim_protein'] = 30
		base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	elif target_encoding == 'Conjoint_triad':
		base_config['input_dim_protein'] = 343
		base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	elif target_encoding == 'Quasi-seq':
		base_config['input_dim_protein'] = 100
		base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
	elif target_encoding == 'CNN':
		base_config['cnn_target_filters'] = cnn_target_filters
		base_config['cnn_target_kernels'] = cnn_target_kernels
	elif target_encoding == 'CNN_RNN':
		base_config['rnn_Use_GRU_LSTM_target'] = rnn_Use_GRU_LSTM_target
		base_config['rnn_target_hid_dim'] = rnn_target_hid_dim
		base_config['rnn_target_n_layers'] = rnn_target_n_layers
		base_config['rnn_target_bidirectional'] = rnn_target_bidirectional 
		base_config['cnn_target_filters'] = cnn_target_filters
		base_config['cnn_target_kernels'] = cnn_target_kernels
	elif target_encoding == 'Transformer':
		base_config['input_dim_protein'] = 4114
		base_config['transformer_emb_size_target'] = transformer_emb_size_target
		base_config['transformer_num_attention_heads_target'] = transformer_num_attention_heads_target
		base_config['transformer_intermediate_size_target'] = transformer_intermediate_size_target
		base_config['transformer_n_layer_target'] = transformer_n_layer_target	
		base_config['transformer_dropout_rate'] = transformer_dropout_rate
		base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
		base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
		base_config['hidden_dim_protein'] = transformer_emb_size_target
	else:
		raise AttributeError("Please use the correct protein encoding available!")

	return base_config

def convert_y_unit(y, from_, to_):
	# basis as nM

	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		y = -np.log10(y*1e-9 + 1e-10)
	elif to_ == 'nM':
		y = y

	return y

def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def drug2emb_encoder(x):

    max_d = 50
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
    '''
		the returned tuple is fed into models.transformer.forward() 
    '''


# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T    

def save_dict(path, obj):
	with open(path + '/config.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
	with open(path + '/config.pkl', 'rb') as f:
		return pickle.load(f)

def download_pretrained_model(model_name, save_dir = './save_folder'):
	if model_name == 'DeepDTA':
		print('Beginning Downloading DeepDTA Model...')
		url = 'https://deeppurpose.s3.amazonaws.com/pretrained_DeepDTA.zip'
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		if not os.path.exists(os.path.join(save_dir, 'pretrained_DeepDTA')):
			os.mkdir(os.path.join(save_dir, 'pretrained_DeepDTA'))

		pretrained_dir = os.path.join(save_dir, 'pretrained_DeepDTA')
		pretrained_dir_ = wget.download(url, pretrained_dir)

		print('Downloading finished... Beginning to extract zip file...')
		with ZipFile(pretrained_dir_, 'r') as zip: 
		    zip.extractall(path = pretrained_dir)
		print('pretrained_DeepDTA Successfully Downloaded...')
		pretrained_dir = os.path.join(pretrained_dir, 'pretrained_DeepDTA')
		return pretrained_dir        