import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, calcPubChemFingerAll
import torch
from torch.utils import data

def smiles2ecfp(s, radius = 2, nBits = 2048):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for ecfp: ' + s + ' convert to all 1 features')
        features = np.ones((nBits, 1))
    return features

def smiles2rdkit2d(s):    
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(s)[1:]
    except:
        print('descriptastorus not found this smiles for ecfp: ' + s + ' convert to all 1 features')
        features = np.ones((200, 1))
    return np.array(features)

# random_fold
def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

# unseen protein
def create_fold_setting_unseen_protein(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['Target Sequence'].isin(gene_drop)]

    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test


# unseen drug
def create_fold_setting_unseen_drug(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['SMILES'].isin(drug_drop)]

    train_val = df[~df['SMILES'].isin(drug_drop)]
    
    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]
    
    return train, val, test


def data_process(X_drug, X_target, y, drug_encoding, target_encoding, split_method = 'random', frac = [0.7, 0.1, 0.2]):
'''
	drug_encoding, select from 
'''
	df_data = pd.DataFrame(zip(X_drug, X_target, y))
	df_data.rename(columns={0:'SMILES',
	                          1: 'Target Sequence',
	                          2: 'Label'}, 
	                 inplace=True)

	if drug_encoding == 'ECFP':
		df_data['drug_encoding'] = df_data.SMILES.apply(smiles2ecfp)
	elif drug_encoding == 'Pubchem':
		df_data['drug_encoding'] = df_data.SMILES.apply(calcPubChemFingerAll)
	elif drug_encoding == 'Path-FP':
		raise NotImplementedError
	elif drug_encoding == 'rdkit_2d_normalized':
		try:
		    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
		 	df_data['drug_encoding'] = df_data.SMILES.apply(smiles2rdkit2d)
		except ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus.")
	elif drug_encoding == 'SMILES_CNN':
		raise NotImplementedError
	elif drug_encoding == 'SMILES_Transformer':
		raise NotImplementedError
	elif drug_encoding == 'MPNN':
		raise NotImplementedError
	else:
		raise NotImplementedError("Please use the correct drug encoding available!")


	if target_encoding == 'AAC':
		df_data['target_encoding'] = df_data['Target Sequence'].apply(CalculateAADipeptideComposition)
	elif target_encoding == 'PseudoAAC':
		df_data['target_encoding'] = df_data['Target Sequence'].apply(_GetPseudoAAC)
	elif target_encoding == 'CNN':
		raise NotImplementedError		
	elif target_encoding == 'Transformer':
		raise NotImplementedError
	else:
		raise NotImplementedError("Please use the correct protein encoding available!")


	if split_method == 'random': 
		train, val, test = create_fold(df_data, np.random.choice(list(range(1000)), 1)[0], frac)
	elif split_method == 'unseen_drug':
		train, val, test = create_fold_setting_unseen_drug(df_data, np.random.choice(list(range(1000)), 1)[0], frac)
	elif split_method == 'unseen_protein':
		train, val, test = create_fold_setting_unseen_protein(df_data, np.random.choice(list(range(1000)), 1)[0], frac)
	elif:
		raise NotImplementedError("Please select one of the three split method: random, unseen_drug, unseen_target!")

	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

	#(train.drug_encoding.values, train.target_encoding.values), (val.drug_encoding.val, train.target_encoding.values), (test.drug_encoding.values, test.target_encoding.values), train.Label.values, val.Label.values, test.Label.values

class data_process_loader(data.Dataset):

    def __init__(self, list_IDs, labels, df, config):
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
        v_p = self.df.iloc[index]['target_encoding']
        v_d = self.df.iloc[index]['drug_encoding']
            
        y = self.labels[index]
        return v_d, v_p, y


