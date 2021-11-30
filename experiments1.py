import DeepPurpose.DeepPurpose.DTI as models
from DeepPurpose.DeepPurpose.utils import *
from DeepPurpose.DeepPurpose.dataset import *
import pickle
from torch.utils.data import SequentialSampler

# X_drug, X_target, y = process_BindingDB(download_BindingDB('../data/BindingDB_All.tsv'),
#                                         y='IC50',
#                                         binary=True,
#                                         convert_to_log=False,
#                                         threshold=30)
#
drug_encoding, target_encoding = 'Morgan', 'AAC'
#
train_under = pd.read_csv('train_under.csv')
train, _, _ = data_process(train_under['SMILES'], train_under['Target Sequence'], train_under['Label'],
                           drug_encoding, target_encoding,
                           split_method='random',
                           frac=[1, 0, 0],
                           random_seed=1)

val_under = pd.read_csv('val_under.csv')
val, _, _ = data_process(val_under['SMILES'], val_under['Target Sequence'], val_under['Label'],
                         drug_encoding, target_encoding,
                         split_method='random',
                         frac=[1, 0, 0],
                         random_seed=1)

test_under = pd.read_csv('test_under.csv')
test, _, _ = data_process(test_under['SMILES'], test_under['Target Sequence'], test_under['Label'],
                          drug_encoding, target_encoding,
                          split_method='random',
                          frac=[1, 0, 0],
                          random_seed=1)

# with open("train.pkl", "rb") as fp:
#     train = pickle.load(fp)
#
# with open("test.pkl", "rb") as fp:
#     test = pickle.load(fp)
#
# with open("val.pkl", "rb") as fp:
#     val = pickle.load(fp)

config = generate_config(drug_encoding,
                         target_encoding,
                         cls_hidden_dims=[1024, 1024, 512],
                         train_epoch=50,
                         LR=0.001,
                         batch_size=256,
                         gnn_num_layers=9)

net = models.model_initialize(**config)

# net.load_pretrained("GCN_AAC.pt/model.pt")
# net.model.to(net.device)


net.train(train=train, val=val, test=test)

net.save_model('Morgan_AAC_under')
#
net.binary = True
info = data_process_loader(test.index.values, test.Label.values, test, **config)

params_test = {'batch_size': 256, 'shuffle': False, 'num_workers': config['num_workers'], 'drop_last': False,
               'sampler': SequentialSampler(info)}

testing_generator = data.DataLoader(info, **params_test)
# #
auc, auprc, f1, loss, y_pred = net.test_(testing_generator, net.model, test=True)

with open("y_label_Morgan_AAC_Under.pkl", 'wb') as fp:
    pickle.dump(test['Label'], fp)

with open("y_pred_Morgan_Under.pkl", 'wb') as fp:
    pickle.dump(y_pred, fp)
