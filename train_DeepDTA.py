from DeepPurpose import utils, dataset
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

drug_encoding = 'CNN'
target_encoding = 'CNN'

# KIBA
#X_drugs, X_targets, y = load_process_KIBA('./data/', binary=False, threshold=9)

# DAVIS
#X_drugs, X_targets, y = load_process_DAVIS('./data/', binary=False)


# BindingDB
data_path = dataset.download_BindingDB('./data/')
X_drugs, X_targets, y = dataset.process_BindingDB(path = data_path, df = None, y = 'Kd', binary = False, convert_to_log = True, threshold = 30)



train, val, test = data_process(X_drugs, X_targets, y, drug_encoding, target_encoding, split_method='random',frac=[0.7,0.1,0.2])

# use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
config = generate_config(drug_encoding = drug_encoding, target_encoding = target_encoding, cls_hidden_dims = [1024,1024,512], train_epoch = 100, LR = 0.001, batch_size = 256, cnn_drug_filters = [32,64,96], cnn_target_filters = [32,64,96], cnn_drug_kernels = [4,6,8], cnn_target_kernels = [4,8,12])

print('There are ' + str(len(X_drugs)) + ' drug-target pairs.')

model = models.model_initialize(**config)
model.train(train, val, test)

#model.save_model('./model/DeepDTA/KIBA/')
#model.save_model('./model/DeepDTA/DAVIS/')
model.save_model('./model/DeepDTA/BindingDB/')
