<h1 align="center">
<p>DeepPurpose
</h1>

<h3 align="center">
<p> A Drug Repurposing and Virtual Screening Toolkit with State-of-the-Art Deep Learning Methods
</h3>

This repository hosts DeepPurpose, a Deep Learning Based Drug Repurposing and Virtual Screening Toolkit (using PyTorch). It allows extremely easy usage (only ten lines of codes) for any non-computational domain researchers to be able to obtain a list of potential drugs using state-of-the-art deep learning while facilitating deep learning method research in this field by providing a flexible framework and baseline. 


### Features

- 10 lines of code from raw data to output drug repurposing/virtual screening result, designed to allow wet-lab biochemists to leverage the power of deep learning and machine learning researchers to push forward the frontiers of DTI prediction. 

- 15+ state-of-the-art encodings for drugs and proteins, ranging from deep neural network on classic cheminformatics fingerprints, CNN-RNN, transformers to message passing graph neural network. Most of the combinations of the encodings are not yet in existing works. Switching encoding is as simple as changing the encoding names!

- Realistic and user-friendly design: 
	- automatic identification to do drug target binding affinity (regression) or drug target interaction prediction (binary) task.
	- support cold target, cold drug settings for robust model evaluations and support single-target high throughput sequencing assay data setup.
	- many dataset loading/downloading/unzipping scripts to ease the tedious preprocessing. 
	- many pretraining checkpoints for popular existing published models.
	- label unit conversion for skewed label distribution such as Kd.
	- time reference for computational expensive encoding.
	- easy monitoring of training process with detailed training metrics output also support early stopping.
	- detailed output records such as rank list for repurposing result.
	- various evaluation metrics: ROC-AUC, PR-AUC, F1 for binary task, MSE, R-squared, Concordance Index for regression task.
	- PyTorch based, support CPU, GPU, Multi-GPUs.
	
	
## Example

### Use Case 1:
Given a new target sequence (e.g. SARS-CoV 3CL Protease), retrieve a list of repurposing drugs.

```python
import DeepPurpose.oneliner as oneliner
oneliner.repurpose(load_SARS_CoV_Protease_3CL())


```


```python
import DeepPurpose.models as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

# Load Data, an array of SMILES for drug, an array of Amino Acid Sequence for Target and an array of binding values/0-1 label.
# e.g. ['Cc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1', ...], ['MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTH...', ...], [0.46, 0.49, ...]
# In this example, BindingDB with Kd binding score is used.
X_drug, X_target, y  = process_BindingDB(download_BindingDB(SAVE_PATH),
					 y = 'Kd', 
					 binary = False, 
					 convert_to_log = True)

# Type in the encoding names for drug/protein.
drug_encoding, target_encoding = 'MPNN', 'Transformer'

# Data processing, here we select cold protein split setup.
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='cold_protein', 
                                frac=[0.7,0.1,0.2])

# Generate new model using default values; also allow model tuning via input parameters.
config = generate_config(drug_encoding, target_encoding, transformer_n_layer_target = 3)
net = models.model_initialize(**config)

# Train the new model.
# Detailed output including a tidy table storing validation loss, metrics, AUC curves figures and etc. are stored in the ./result folder.
net.train(train, val, test)

# or simply load pretrained model from a model directory path
net = models.model_pretrained(MODEL_PATH_DIR)

# Repurpose using the trained model or pre-trained model
# In this example, loading repurposing dataset using Broad Repurposing Hub and SARS-CoV 3CL Protease Target.
X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
target, target_name = load_SARS_CoV_Protease_3CL()

_ = models.repurpose(X_repurpose, target, net, drug_name, target_name)
'''
Output:
------------------
Drug Repurposing Result for SARS-CoV 3CL Protease
Drug Rufloxacin     predicted to have binding affinity score 4.87
Drug Sparfloxacin   predicted to have binding affinity score 5.39
Drug Cefoperazone   predicted to have binding affinity score 4.70
...
'''

# Virtual screening using the trained model or pre-trained model 
# In this example, model is trained with binary outcome and customized input is provided. 
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

_ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)
'''
Output:
------------------
Virtual Screening Result
Drug 16007391   predicted to NOT have interaction with the target P36896 with interaction probablity of 0.23
Drug 44355753   predicted to have interaction with the target P00374 with interaction probablity of 0.71
Drug 24180719   predicted to NOT have interaction with the target P61075 with interaction probablity of 0.31
'''
```

## Install & Usage
```bash
# -- First Time -- #
git clone git@github.com:kexinhuang12345/DeepPurpose.git
# Download code repository

cd DeepPurpose
# Change directory to DeepPurpose

conda env create -f environ.yml  
## Build virtual environment with all packages installed using conda

conda activate deeppurpose
## Activate conda environment

## run our code
... ...

conda deactivate 
## Exit conda environment 

# -- In the future -- #
cd DeepPurpose
# Change directory to DeepPurpose

conda activate deeppurpose
## Activate conda environment

## run our code
... ...

conda deactivate 
## Exit conda environment 
```

Checkout demos & tutorials in the [DEMO](https://github.com/kexinhuang12345/DeepPurpose/tree/master/DEMO) folder to start...

## Encodings
Currently, we support the following encodings:

| Drug Encodings  | Description |
|-----------------|-------------|
| Morgan | Extended-Connectivity Fingerprints |
| Pubchem| Pubchem Substructure-based Fingerprints|
| Daylight | Daylight-type fingerprints | 
| rdkit_2d_normalized| Normalized Descriptastorus|
| CNN | Convolutional Neural Network on SMILES|
|CNN_RNN| A GRU/LSTM on top of a CNN on SMILES|
|Transformer| Transformer Encoder on ESPF|
|  MPNN | Message-passing neural network |

| Target Encodings  | Description |
|-----------------|-------------|
| AAC | Amino acid composition up to 3-mers |
| PseudoAAC| Pseudo amino acid composition|
| Conjoint_triad | Conjoint triad features | 
| Quasi-seq| Quasi-sequence order descriptor|
| CNN | Convolutional Neural Network on target seq|
|CNN_RNN| A GRU/LSTM on top of a CNN on target seq|
|Transformer| Transformer Encoder on ESPF|

## Documentations


## Cite Us

Please cite [arxiv]():
```
@article{
}

```

## Contact
Please contact kexinhuang@hsph.harvard.edu or tfu42@gatech.edu for help or submit an issue. 



