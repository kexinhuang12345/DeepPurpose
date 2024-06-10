import pandas as pd
import numpy as np
import wget
from zipfile import ZipFile
from DeepPurpose.utils import *
import json
import os
import requests
import re

'''
Acknowledgement:
The BindingDB dataset is hosted in https://www.bindingdb.org/bind/index.jsp.

The Davis Dataset can be found in http://staff.cs.utu.fi/~aatapa/data/DrugTarget/.

The KIBA dataset can be found in https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z.

The Drug Target Common Dataset can be found in https://drugtargetcommons.fimm.fi/.

The COVID-19 Dataset including SARS-CoV, Broad Repurposing Hub can be found in https://www.aicures.mit.edu/data; and https://pubchem.ncbi.nlm.nih.gov/bioassay/1706.
We use some existing files from https://github.com/yangkevin2/coronavirus_data

We use the SMILES, protein sequence from DeepDTA github repo: https://github.com/hkmztrk/DeepDTA/tree/master/data.
'''

def read_file_training_dataset_bioassay(path):
	# a line in the file is SMILES score, the first line is the target sequence
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	target = file.readline()
	if target[-1:] == '\n':
		target = target[:-1]
	X_drug = []
	y = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[0])
		y.append(float(values[1]))
	file.close()
	return np.array(X_drug), target, np.array(y)

def read_file_training_dataset_drug_target_pairs(path):
	# a line in the file is SMILES Target_seq score
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_target = []
	y = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[0])
		X_target.append(values[1])
		y.append(float(values[2]))
	file.close()
	return np.array(X_drug), np.array(X_target), np.array(y)

def read_file_training_dataset_drug_drug_pairs(path):
	# a line in the file is SMILES SMILES score
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_target = []
	y = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[0])
		X_target.append(values[1])
		y.append(float(values[2]))
	file.close()
	return np.array(X_drug), np.array(X_target), np.array(y)

def read_file_protein_function(path):
	# a line in the file is protein names and target_seq
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_drug_names = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[1])
		X_drug_names.append(values[0])
	file.close()
	return np.array(X_drug), np.array(X_drug_names)

def read_file_compound_property(path):
	# a line in the file is drug names and smiles
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_drug_names = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[1])
		X_drug_names.append(values[0])
	file.close()
	return np.array(X_drug), np.array(X_drug_names)

def read_file_training_dataset_protein_protein_pairs(path):
	# a line in the file is target_seq target_seq score
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_target = []
	y = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[0])
		X_target.append(values[1])
		y.append(float(values[2]))
	file.close()
	return np.array(X_drug), np.array(X_target), np.array(y)

def read_file_virtual_screening_drug_target_pairs(path):
	# a line in the file is SMILES Target_seq
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_target = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[0])
		X_target.append(values[1])
	file.close()
	return np.array(X_drug), np.array(X_target)


def read_file_repurposing_library(path):
	# a line in the file is drug names and SMILES
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	X_drug = []
	X_drug_names = []
	for aline in file:
		values = aline.split()
		X_drug.append(values[1])
		X_drug_names.append(values[0])
	file.close()
	return np.array(X_drug), np.array(X_drug_names)

def read_file_target_sequence(path):
	# a line in the file is target name and target sequence
	try:
		file = open(path, "r")
	except:
		print('Path Not Found, please double check!')
	values = file.readline().split()
	file.close()
	return values[1], values[0]


def download_BindingDB(path = './data'):

	print('Beginning to download dataset...')

	if not os.path.exists(path):
	    os.makedirs(path)

	try:
	    url = "https://www.bindingdb.org/bind/downloads/" + [url.split('/')[-1] for url in re.findall(
		    r'(/rwd/bind/chemsearch/marvin/SDFdownload.jsp\?download_file=/bind/downloads/BindingDB_All_.*?\.tsv\.zip)',
			requests.get("https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp").text)][0]
	except Exception:
	    print("Failed to retrieve current URL for BindingDB, falling back on hard-coded URL")
	    url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_202406_tsv.zip"
	saved_path = wget.download(url, path)

	print('Beginning to extract zip file...')
	with ZipFile(saved_path, 'r') as zip:
	    zip.extractall(path = path)
	    print('Done!')
	path = path + '/BindingDB_All_202406.tsv'
	return path


def download_DrugTargetCommons(path):

	print('Beginning to download dataset...')

	if not os.path.exists(path):
	    os.makedirs(path)

	url = 'https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv'
	saved_path = wget.download(url, path)
	path = path + '/DtcDrugTargetInteractions.csv'
	return path

def process_BindingDB(path = None, df = None, y = 'Kd', binary = False, \
					convert_to_log = True, threshold = 30, return_ids = False, \
					ids_condition = 'OR', harmonize_affinities = None):
	"""
	:path: path to original BindingDB CSV/TSV data file. If None, then 'df' is expected.
	:param df: pre-loaded DataFrame
	:param y: type of binding affinity label. can be either 'Kd', 'IC50', 'EC50', 'Ki',
				or a list of strings with multiple choices.
	:param binary: whether to use binary labels
	:param convert_to_log: whether to convert nM units to P (log)
	:param threshold: threshold affinity for binary labels. can be a number or list
				of two numbers (low and high threshold)
	:param return_ids: whether to return drug and target ids
	:param ids_condition: keep samples for which drug AND/OR target IDs exist
	:param harmonize_affinities:  unify duplicate samples
							'max' for choosing the instance with maximum affinity
							'mean' for using the mean affinity of all duplicate instances
							None to do nothing
	"""	
	if path is not None and not os.path.exists(path):
		os.makedirs(path)

	if df is not None:
		print('Loading Dataset from the pandas input...')
	elif path is not None:
		print('Loading Dataset from path...')
		df = pd.read_csv(path, sep = '\t', error_bad_lines=False)
	else:
		ValueError("Either 'df' of 'path' must be provided")

	print('Beginning Processing...')
	df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
	df = df[df['Ligand SMILES'].notnull()]

	idx_str = []
	yy = y
	if isinstance(y, str):
		yy = [y]
	for y in yy:
		if y == 'Kd':
			idx_str.append('Kd (nM)')
		elif y == 'IC50':
			idx_str.append('IC50 (nM)')
		elif y == 'Ki':
			idx_str.append('Ki (nM)')
		elif y == 'EC50':
			idx_str.append('EC50 (nM)')
		else:
			print('select Kd, Ki, IC50 or EC50')
	
	if len(idx_str)==1:
		df_want = df[df[idx_str[0]].notnull()]
	else: # select multiple affinity measurements.                 
		# keep rows for which at least one of the columns in the idx_str list is not null
		df_want = df.dropna(thresh=1, subset=idx_str) 
		
	df_want = df_want[['BindingDB Reactant_set_id', 'Ligand InChI', 'Ligand SMILES',\
					'PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain',\
					'BindingDB Target Chain Sequence'] + idx_str]
	
	for y in idx_str:
		df_want[y] = df_want[y].str.replace('>', '')
		df_want[y] = df_want[y].str.replace('<', '')
		df_want[y] = df_want[y].astype(float)
	
	# Harmonize into single label using the mean of existing labels:
	df_want['Label'] = df_want[idx_str].mean(axis=1, skipna=True)

	df_want.rename(columns={'BindingDB Reactant_set_id':'ID',
							'Ligand SMILES':'SMILES',
							'Ligand InChI':'InChI',
							'PubChem CID':'PubChem_ID',
							'UniProt (SwissProt) Primary ID of Target Chain':'UniProt_ID',
							'BindingDB Target Chain Sequence': 'Target Sequence'},
							inplace=True)

	# have at least uniprot or pubchem ID
	if ids_condition == 'OR':
		df_want = df_want[df_want.PubChem_ID.notnull() | df_want.UniProt_ID.notnull()]
	elif ids_condition == 'AND':
		df_want = df_want[df_want.PubChem_ID.notnull() & df_want.UniProt_ID.notnull()]
	else:
		ValueError("ids_condition must be set to 'OR' or 'AND'")

	df_want = df_want[df_want.InChI.notnull()]

	df_want = df_want[df_want.Label <= 10000000.0]
	print('There are ' + str(len(df_want)) + ' drug target pairs.')

	if harmonize_affinities is not None:
		df_want = df_want[['PubChem_ID', 'SMILES', 'UniProt_ID', 'Target Sequence', 'Label']]
		if harmonize_affinities.lower() == 'max_affinity':
			df_want = df_want.groupby(['PubChem_ID', 'SMILES', 'UniProt_ID', 'Target Sequence']).Label.agg(min).reset_index()
		if harmonize_affinities.lower() == 'mean':
			df_want = df_want.groupby(['PubChem_ID', 'SMILES', 'UniProt_ID', 'Target Sequence']).Label.agg(np.mean).reset_index()
	
	if binary:
		print('Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter')
		if isinstance(threshold, Sequence):
			# filter samples with affinity values between the thresholds
			df_want = df_want[(df_want.Label < threshold[0]) | (df_want.Label > threshold[1])]
		else: # single threshold
			threshold = [threshold]
		y = [1 if i else 0 for i in df_want.Label.values < threshold[0]]
	else:
		if convert_to_log:
			print('Default set to logspace (nM -> p) for easier regression')
			y = convert_y_unit(df_want.Label.values, 'nM', 'p')
		else:
			y = df_want.Label.values
		
	if return_ids:
		return df_want.SMILES.values, df_want['Target Sequence'].values, np.array(y), df_want['PubChem_ID'].values, df_want['UniProt_ID'].values
	return df_want.SMILES.values, df_want['Target Sequence'].values, np.array(y)


def load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30):
	print('Beginning Processing...')

	if not os.path.exists(path):
	    os.makedirs(path)

	# url = 'https://drive.google.com/uc?export=download&id=14h-0YyHN8lxuc0KV3whsaSaA-4KSmiVN'
	url = 'https://github.com/futianfan/DeepPurpose_Data/blob/main/DAVIS.zip?raw=true'
	saved_path = wget.download(url, path)

	print('Beginning to extract zip file...')
	with ZipFile(saved_path, 'r') as zip:
	    zip.extractall(path = path)

	affinity = pd.read_csv(path + '/DAVIS/affinity.txt', header=None, sep = ' ')

	with open(path + '/DAVIS/target_seq.txt') as f:
		target = json.load(f)

	with open(path + '/DAVIS/SMILES.txt') as f:
		drug = json.load(f)

	target = list(target.values())
	drug = list(drug.values())

	SMILES = []
	Target_seq = []
	y = []

	for i in range(len(drug)):
		for j in range(len(target)):
			SMILES.append(drug[i])
			Target_seq.append(target[j])
			y.append(affinity.values[i, j])

	if binary:
		print('Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter')
		y = [1 if i else 0 for i in np.array(y) < threshold]
	else:
		if convert_to_log:
			print('Default set to logspace (nM -> p) for easier regression')
			y = convert_y_unit(np.array(y), 'nM', 'p')
		else:
			y = y
	print('Done!')
	return np.array(SMILES), np.array(Target_seq), np.array(y)

def load_process_KIBA(path = './data', binary = False, threshold = 9):
	print('Beginning Processing...')


	if not os.path.exists(path):
	    os.makedirs(path)

	# url = 'https://drive.google.com/uc?export=download&id=1fb3ZI-3_865OuRMWNMzLPnbLm9CktM44'
	url = 'https://github.com/futianfan/DeepPurpose_Data/blob/main/KIBA.zip?raw=true'
	saved_path = wget.download(url, path)

	print('Beginning to extract zip file...')
	with ZipFile(saved_path, 'r') as zip:
	    zip.extractall(path = path)

	affinity = pd.read_csv(path + '/KIBA/affinity.txt', header=None, sep = '\t')
	affinity = affinity.fillna(-1)

	with open(path + '/KIBA/target_seq.txt') as f:
		target = json.load(f)

	with open(path + '/KIBA/SMILES.txt') as f:
		drug = json.load(f)

	target = list(target.values())
	drug = list(drug.values())

	SMILES = []
	Target_seq = []
	y = []

	for i in range(len(drug)):
		for j in range(len(target)):
			if affinity.values[i, j] != -1:
				SMILES.append(drug[i])
				Target_seq.append(target[j])
				y.append(affinity.values[i, j])

	if binary:
		print('Note that KIBA is not suitable for binary classification as it is a modified score. \
			   Default binary threshold for the binding affinity scores are 9, \
			   you should adjust it by using the "threshold" parameter')
		y = [1 if i else 0 for i in np.array(y) < threshold]
	else:
		y = y

	print('Done!')
	return np.array(SMILES), np.array(Target_seq), np.array(y)

def load_AID1706_SARS_CoV_3CL(path = './data', binary = True, threshold = 15, balanced = True, oversample_num = 30, seed = 1):
	print('Beginning Processing...')

	if not os.path.exists(path):
		os.makedirs(path)

	target = 'SGFKKLVSPSSAVEKCIVSVSYRGNNLNGLWLGDSIYCPRHVLGKFSGDQWGDVLNLANNHEFEVVTQNGVTLNVVSRRLKGAVLILQTAVANAETPKYKFVKANCGDSFTIACSYGGTVIGLYPVTMRSNGTIRASFLAGACGSVGFNIEKGVVNFFYMHHLELPNALHTGTDLMGEFYGGYVDEEVAQRVPPDNLVTNNIVAWLYAAIISVKESSFSQPKWLESTTVSIEDYNRWASDNGFTPFSTSTAITKLSAITGVDVCKLLRTIMVKSAQWGSDPILGQYNFEDELTPESVFNQVGGVRLQ'
	url = 'https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid=1706'
	saved_path_data = wget.download(url, path)

	# url = 'https://drive.google.com/uc?export=download&id=1eipPaFrg-mVULoBhyp2kvEemi2WhDxsM'
	url = 'https://github.com/futianfan/DeepPurpose_Data/blob/main/AID1706_training_conversions.csv?raw=true'
	saved_path_conversion = wget.download(url, path)

	df_data = pd.read_csv(saved_path_data)
	df_conversion = pd.read_csv(saved_path_conversion)
	val = df_data.iloc[4:][['PUBCHEM_CID','PUBCHEM_ACTIVITY_SCORE']]

	val['binary_label'] = 0
	val['binary_label'][(val.PUBCHEM_ACTIVITY_SCORE >= threshold) & (val.PUBCHEM_ACTIVITY_SCORE <=100)] = 1

	if balanced:
		val = pd.concat([val[val.binary_label==0].sample(n = len(val[val.binary_label==1]) * oversample_num, replace = False, random_state = seed), pd.concat([val[val.binary_label==1]]*oversample_num, ignore_index=True)]).sample(frac = 1, replace = False, random_state = seed).reset_index(drop = True)

	cid2smiles = dict(zip(df_conversion[['cid','smiles']].values[:, 0], df_conversion[['cid','smiles']].values[:, 1]))
	X_drug = [cid2smiles[i] for i in val.PUBCHEM_CID.values]

	if binary:
		print('Default binary threshold for the binding affinity scores is 15, recommended by the investigator')
		y = val.binary_label.values
	else:
		y = val.PUBCHEM_ACTIVITY_SCORE.values

	print('Done!')
	return np.array(X_drug), target, np.array(y)

def load_HIV(path = './data'):
	download_unzip('HIV', path, 'hiv.csv')

	df = pd.read_csv(os.path.join(path,'HIV.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]

	df = df[df["HIV_active"].notnull()].reset_index(drop = True)
	y = df["HIV_active"].values
	drugs = df.smiles.values
	drugs_idx = np.array(list(range(len(drugs))))

	return drugs, y, drugs_idx

def load_AqSolDB(path = './data'):

	if os.path.exists(os.path.join(path,'curated-solubility-dataset.csv')):
		print('Dataset already downloaded in the local system...', flush = True, file = sys.stderr)
	else:
		wget.download('https://dataverse.harvard.edu/api/access/datafile/3407241?format=original&gbrecs=true', path)

	df = pd.read_csv(os.path.join(path,'curated-solubility-dataset.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df["Solubility"].values
	drugs = df.SMILES.values
	drugs_idx = df.Name.values

	return drugs, y, drugs_idx

def load_broad_repurposing_hub(path = './data'):
	url = 'https://dataverse.harvard.edu/api/access/datafile/4159648'
	if not os.path.exists(path):
	    os.makedirs(path)
	download_path = os.path.join(path, 'broad.tab')
	download_url(url, download_path)
	df = pd.read_csv(download_path, sep = '\t')
	df = df.fillna('UNK')
	return df.smiles.values, df.title.values, df.cid.values.astype(str)

def load_antiviral_drugs(path = './data', no_cid = False):
	url = 'https://dataverse.harvard.edu/api/access/datafile/4159652'
	if not os.path.exists(path):
	    os.makedirs(path)
	download_path = os.path.join(path, 'antiviral_drugs.tab')
	download_url(url, download_path)
	df = pd.read_csv(download_path, sep = '\t')
	if no_cid:
		return df.SMILES.values, df[' Name'].values
	else:
		return df.SMILES.values, df[' Name'].values, df['Pubchem CID'].values

def load_IC50_Not_Pretrained(path = './data', n=500):
	print('Downloading...')
	url = 'https://dataverse.harvard.edu/api/access/datafile/4159695'
	if not os.path.exists(path):
	    os.makedirs(path)
	download_path = os.path.join(path, 'IC50_not_Kd.csv')
	download_url(url, download_path)
	df = pd.read_csv(download_path).sample(n = n, replace = False).reset_index(drop = True)
	return df['Target Sequence'].values, df['SMILES'].values

def load_IC50_1000_Samples(path = './data', n=100):
	print('Downloading...')
	url = 'https://dataverse.harvard.edu/api/access/datafile/4159681'
	if not os.path.exists(path):
	    os.makedirs(path)
	download_path = os.path.join(path, 'IC50_samples.csv')
	download_url(url, download_path)
	df = pd.read_csv(download_path).sample(n = n, replace = False).reset_index(drop = True)
	return df['Target Sequence'].values, df['SMILES'].values

def load_SARS_CoV_Protease_3CL():
	target = 'SGFKKLVSPSSAVEKCIVSVSYRGNNLNGLWLGDSIYCPRHVLGKFSGDQWGDVLNLANNHEFEVVTQNGVTLNVVSRRLKGAVLILQTAVANAETPKYKFVKANCGDSFTIACSYGGTVIGLYPVTMRSNGTIRASFLAGACGSVGFNIEKGVVNFFYMHHLELPNALHTGTDLMGEFYGGYVDEEVAQRVPPDNLVTNNIVAWLYAAIISVKESSFSQPKWLESTTVSIEDYNRWASDNGFTPFSTSTAITKLSAITGVDVCKLLRTIMVKSAQWGSDPILGQYNFEDELTPESVFNQVGGVRLQ'
	target_name = 'SARS-CoV 3CL Protease'
	return target, target_name

def load_SARS_CoV2_Protease_3CL():
	target = 'SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ'
	target_name = 'SARS-CoV2 3CL Protease'
	return target, target_name

def load_LCK():
	target = 'MGCGCSSHPEDDWMENIDVCENCHYPIVPLDGKGTLLIRNGSEVRDPLVTYEGSNPPASPLQDNLVIALHSYEPSHDGDLGFEKGEQLRILEQSGEWWKAQSLTTGQEGFIPFNFVAKANSLEPEPWFFKNLSRKDAERQLLAPGNTHGSFLIRESESTAGSFSLSVRDFDQNQGEVVKHYKIRNLDNGGFYISPRITFPGLHELVRHYTNASDGLCTRLSRPCQTQKPQKPWWEDEWEVPRETLKLVERLGAGQFGEVWMGYYNGHTKVAVKSLKQGSMSPDAFLAEANLMKQLQHQRLVRLYAVVTQEPIYIITEYMENGSLVDFLKTPSGIKLTINKLLDMAAQIAEGMAFIEERNYIHRDLRAANILVSDTLSCKIADFGLARLIEDNEYTAREGAKFPIKWTAPEAINYGTFTIKSDVWSFGILLTEIVTHGRIPYPGMTNPEVIQNLERGYRMVRPDNCPEELYQLMRLCWKERPEDRPTFDYLRSVLEDFFTATEGQYQPQP'
	target_name = 'Tyrosine-protein kinase Lck'
	return target, target_name

def load_SARS_CoV2_RNA_polymerase():
	target = 'SADAQSFLNRVCGVSAARLTPCGTGTSTDVVYRAFDIYNDKVAGFAKFLKTNCCRFQEKDEDDNLIDSYFVVKRHTFSNYQHEETIYNLLKDCPAVAKHDFFKFRIDGDMVPHISRQRLTKYTMADLVYALRHFDEGNCDTLKEILVTYNCCDDDYFNKKDWYDFVENPDILRVYANLGERVRQALLKTVQFCDAMRNAGIVGVLTLDNQDLNGNWYDFGDFIQTTPGSGVPVVDSYYSLLMPILTLTRALTAESHVDTDLTKPYIKWDLLKYDFTEERLKLFDRYFKYWDQTYHPNCVNCLDDRCILHCANFNVLFSTVFPPTSFGPLVRKIFVDGVPFVVSTGYHFRELGVVHNQDVNLHSSRLSFKELLVYAADPAMHAASGNLLLDKRTTCFSVAALTNNVAFQTVKPGNFNKDFYDFAVSKGFFKEGSSVELKHFFFAQDGNAAISDYDYYRYNLPTMCDIRQLLFVVEVVDKYFDCYDGGCINANQVIVNNLDKSAGFPFNKWGKARLYYDSMSYEDQDALFAYTKRNVIPTITQMNLKYAISAKNRARTVAGVSICSTMTNRQFHQKLLKSIAATRGATVVIGTSKFYGGWHNMLKTVYSDVENPHLMGWDYPKCDRAMPNMLRIMASLVLARKHTTCCSLSHRFYRLANECAQVLSEMVMCGGSLYVKPGGTSSGDATTAYANSVFNICQAVTANVNALLSTDGNKIADKYVRNLQHRLYECLYRNRDVDTDFVNEFYAYLRKHFSMMILSDDAVVCFNSTYASQGLVASIKNFKSVLYYQNNVFMSEAKCWTETDLTKGPHEFCSQHTMLVKQGDDYVYLPYPDPSRILGAGCFVDDIVKTDGTLMIERFVSLAIDAYPLTKHPNQEYADVFHLYLQYIRKLHDELTGHMLDMYSVMLTNDNTSRYWEPEFYEAMYTPHTVLQ'
	target_name = 'RNA_polymerase_SARS_CoV2'
	return target, target_name

def load_SARS_CoV2_Helicase():
	target = 'AVGACVLCNSQTSLRCGACIRRPFLCCKCCYDHVISTSHKLVLSVNPYVCNAPGCDVTDVTQLYLGGMSYYCKSHKPPISFPLCANGQVFGLYKNTCVGSDNVTDFNAIATCDWTNAGDYILANTCTERLKLFAAETLKATEETFKLSYGIATVREVLSDRELHLSWEVGKPRPPLNRNYVFTGYRVTKNSKVQIGEYTFEKGDYGDAVVYRGTTTYKLNVGDYFVLTSHTVMPLSAPTLVPQEHYVRITGLYPTLNISDEFSSNVANYQKVGMQKYSTLQGPPGTGKSHFAIGLALYYPSARIVYTACSHAAVDALCEKALKYLPIDKCSRIIPARARVECFDKFKVNSTLEQYVFCTVNALPETTADIVVFDEISMATNYDLSVVNARLRAKHYVYIGDPAQLPAPRTLLTKGTLEPEYFNSVCRLMKTIGPDMFLGTCRRCPAEIVDTVSALVYDNKLKAHKDKSAQCFKMFYKGVITHDVSSAINRPQIGVVREFLTRNPAWRKAVFISPYNSQNAVASKILGLPTQTVDSSQGSEYDYVIFTQTTETAHSCNVNRFNVAITRAKVGILCIMSDRDLYDKLQFTSLEIPRRNVATLQ'
	target_name = 'SARS_CoV2_Helicase'
	return target, target_name

def load_SARS_CoV2_3to5_exonuclease():
	target = 'AENVTGLFKDCSKVITGLHPTQAPTHLSVDTKFKTEGLCVDIPGIPKDMTYRRLISMMGFKMNYQVNGYPNMFITREEAIRHVRAWIGFDVEGCHATREAVGTNLPLQLGFSTGVNLVAVPTGYVDTPNNTDFSRVSAKPPPGDQFKHLIPLMYKGLPWNVVRIKIVQMLSDTLKNLSDRVVFVLWAHGFELTSMKYFVKIGPERTCCLCDRRATCFSTASDTYACWHHSIGFDYVYNPFMIDVQQWGFTGNLQSNHDLYCQVHGNAHVASCDAIMTRCLAVHECFVKRVDWTIEYPIIGDELKINAACRKVQHMVVKAALLADKFPVLHDIGNPKAIKCVPQADVEWKFYDAQPCSDKAYKIEELFYSYATHSDKFTDGVCLFWNCNVDRYPANSIVCRFDTRVLSNLNLPGCDGGSLYVNKHAFHTPAFDKSAFVNLKQLPFFYYSDSPCESHGKQVVSDIDYVPLKSATCITRCNLGGAVCRHHANEYRLYLDAYNMMISAGFSLWVYKQFDTYNLWNTFTRLQ'
	target_name = 'SARS_CoV2_3to5_exonuclease'
	return target, target_name

def load_SARS_CoV2_endoRNAse():
	target = 'SLENVAFNVVNKGHFDGQQGEVPVSIINNTVYTKVDGVDVELFENKTTLPVNVAFELWAKRNIKPVPEVKILNNLGVDIAANTVIWDYKRDAPAHISTIGVCSMTDIAKKPTETICAPLTVFFDGRVDGQVDLFRNARNGVLITEGSVKGLQPSVGPKQASLNGVTLIGEAVKTQFNYYKKVDGVVQQLPETYFTQSRNLQEFKPRSQMEIDFLELAMDEFIERYKLEGYAFEHIVYGDFSHSQLGGLHLLIGLAKRFKESPFELEDFIPMDSTVKNYFITDAQTGSSKCVCSVIDLLLDDFVEIIKSQDLSVVSKVVKVTIDYTEISFMLWCKDGHVETFYPKLQ'
	target_name = 'SARS_CoV2_endoRNAse'
	return target, target_name

def load_SARS_CoV2_2_O_ribose_methyltransferase():
	target = 'SSQAWQPGVAMPNLYKMQRMLLEKCDLQNYGDSATLPKGIMMNVAKYTQLCQYLNTLTLAVPYNMRVIHFGAGSDKGVAPGTAVLRQWLPTGTLLVDSDLNDFVSDADSTLIGDCATVHTANKWDLIISDMYDPKTKNVTKENDSKEGFFTYICGFIQQKLALGGSVAIKITEHSWNADLYKLMGHFAWWTAFVTNVNASSSEAFLIGCNYLGKPREQIDGYVMHANYIFWRNTNPIQLSSYSLFDMSKFPLKLRGTAVMSLKEGQINDMILSLLSKGRLIIRENNRVVISSDVLVNN'
	target_name = 'SARS_CoV2_2_O_ribose_methyltransferase'
	return target, target_name

def load_SLC6A2():
	target = 'MLLARMNPQVQPENNGADTGPEQPLRARKTAELLVVKERNGVQCLLAPRDGDAQPRETWGKKIDFLLSVVGFAVDLANVWRFPYLCYKNGGGAFLIPYTLFLIIAGMPLFYMELALGQYNREGAATVWKICPFFKGVGYAVILIALYVGFYYNVIIAWSLYYLFSSFTLNLPWTDCGHTWNSPNCTDPKLLNGSVLGNHTKYSKYKFTPAAEFYERGVLHLHESSGIHDIGLPQWQLLLCLMVVVIVLYFSLWKGVKTSGKVVWITATLPYFVLFVLLVHGVTLPGASNGINAYLHIDFYRLKEATVWIDAATQIFFSLGAGFGVLIAFASYNKFDNNCYRDALLTSSINCITSFVSGFAIFSILGYMAHEHKVNIEDVATEGAGLVFILYPEAISTLSGSTFWAVVFFVMLLALGLDSSMGGMEAVITGLADDFQVLKRHRKLFTFGVTFSTFLLALFCITKGGIYVLTLLDTFAAGTSILFAVLMEAIGVSWFYGVDRFSNDIQQMMGFRPGLYWRLCWKFVSPAFLLFVVVVSIINFKPLTYDDYIFPPWANWVGWGIALSSMVLVPIYVIYKFLSTQGSLWERLAYGITPENEHHLVAQRDIRQFQLQHWLAI'
	target_name = 'SLC6A2'
	return target, target_name

def load_MMP9():
	target = 'MSLWQPLVLVLLVLGCCFAAPRQRQSTLVLFPGDLRTNLTDRQLAEEYLYRYGYTRVAEMRGESKSLGPALLLLQKQLSLPETGELDSATLKAMRTPRCGVPDLGRFQTFEGDLKWHHHNITYWIQNYSEDLPRAVIDDAFARAFALWSAVTPLTFTRVYSRDADIVIQFGVAEHGDGYPFDGKDGLLAHAFPPGPGIQGDAHFDDDELWSLGKGVVVPTRFGNADGAACHFPFIFEGRSYSACTTDGRSDGLPWCSTTANYDTDDRFGFCPSERLYTQDGNADGKPCQFPFIFQGQSYSACTTDGRSDGYRWCATTANYDRDKLFGFCPTRADSTVMGGNSAGELCVFPFTFLGKEYSTCTSEGRGDGRLWCATTSNFDSDKKWGFCPDQGYSLFLVAAHEFGHALGLDHSSVPEALMYPMYRFTEGPPLHKDDVNGIRHLYGPRPEPEPRPPTTTTPQPTAPPTVCPTGPPTVHPSERPTAGPTGPPSAGPTGPPTAGPSTATTVPLSPVDDACNVNIFDAIAEIGNQLYLFKDGKYWRFSEGRGSRPQGPFLIADKWPALPRKLDSVFEERLSKKLFFFSGRQVWVYTGASVLGPRRLDKLGLGADVAQVTGALRSGRGKMLLFSGRRLWRFDVKAQMVDPRSASEVDRMFPGVPLDTHDVFQYREKAYFCQDRFYWRVSSRSELNQVDQVGYVTYDILQCPED'
	target_name = 'MMP9'
	return target, target_name
