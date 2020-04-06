import models 
from utils import *
from dataset import *

X_drug, X_target, y  = process_BindingDB('./data/BindingDB_All.tsv',
                                         y = 'Kd', 
                                         binary = False, 
                                         convert_to_log = True)

drug_encoding = 'rdkit_2d_normalized'
target_encoding = 'AAC'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2])

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 1, 
                         LR = 0.001, 
                         batch_size = 16,
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12],
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3
                        )
model = models.model_initialize(**config)
model.train(train, val, test)
model.save_model('./model_rdkit_AAC')
