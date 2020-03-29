import os
os.chdir('../')

import pandas as pd
import numpy as np

import models as models
from utils import data_process, convert_y_unit, generate_config

df_Kd = pd.read_csv('./Kd/data.csv')
df_Kd = df_Kd.sample(frac = 0.002, replace = False)# toy dataset

X_drug = df_Kd.SMILES.values
X_target = df_Kd['Target Sequence'].values 
# support nM to p (logspace) convertion to help regression
y = convert_y_unit(df_Kd.Kd.values, 'nM', 'p') 

drug_encoding = 'Daylight'
drug_encoding = 'MPNN'
target_encoding = 'Conjoint_triad'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2])



# model setup, you can adjust the config file by typing in model parameters. e.g. cls_hidden_dim = [256, 32]
config = generate_config(drug_encoding, target_encoding, train_epoch = 3)
model = models.model_initialize(**config)


model.train(train, val, test)


test = df_Kd.sample(n = 20, replace=False)
target = test['Target Sequence'].iloc[0]
X_repurpose = test.SMILES.values
#print(type(X_repurpose)) ### np.array shape: (20,)
#print(X_repurpose.shape)

drug_name = test.PubChem_ID.astype(int).astype(str).values
target_name = test.UniProt_ID.iloc[0]

r = models.repurpose(X_repurpose, target, model, drug_name, target_name)

target = test['Target Sequence'].values
target_name = test.UniProt_ID.astype(str).values

r = models.virtual_screening(X_repurpose, target, model, drug_name, target_name)







