<p align="center"><img src="https://github.com/kexinhuang12345/DeepPurpose/blob/master/docs/figs/logo_deeppurpose_horizontal.pdf?raw=true" alt="logo" width="400px" /></p>


<h3 align="center">
<p> A Drug Repurposing and Virtual Screening Toolkit with State-of-the-Art Deep Learning Methods
</h3>

This repository hosts DeepPurpose, a Deep Learning Based Drug Repurposing and Virtual Screening Toolkit (using PyTorch). It allows extremely easy usage (only one line of code!) for any non-computational domain researchers to be able to obtain a list of potential drugs using state-of-the-art deep learning while facilitating deep learning method research in this field by providing a flexible framework (less than 10 lines of codes!) and baselines. 


### Features

- For biomedical researchers, ONE line of code from raw data to output drug repurposing/virtual screening result, designed to allow wet-lab biochemists to leverage the power of deep learning. The result is ensembled from six pretrained models!

- For computational researchers, 15+ state-of-the-art encodings for drugs and proteins, ranging from deep neural network on classic cheminformatics fingerprints, CNN-RNN, transformers to message passing graph neural network, with 50+ models! Most of the combinations of the encodings are not yet in existing works. All of these under 10 lines but with lots of flexibility! Switching encoding is as simple as changing the encoding names!

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
	
Note: We are actively looking for constructive advices/user feedbacks/experiences on using DeepPurpose! Please open an issue or [contact us](kexinhuang@hsph.harvard.edu).
	

## Example

### Case Study 1:
Given a new target sequence (e.g. SARS-CoV 3CL Protease), retrieve a list of repurposing drugs. Results aggregated from six pretrained model!

```python
import DeepPurpose.oneliner as oneliner
oneliner.repurpose(load_SARS_CoV_Protease_3CL())
```

### Case Study 2:
Given a new target sequence (e.g. SARS-CoV 3CL Protease), but training on new data (AID1706 Bioassay), and then retrieve a list of repurposing drugs. The model is finetuned from the pretraining checkpoint!


```python
import DeepPurpose.oneliner as oneliner
oneliner.repurpose(load_SARS_CoV_Protease_3CL(), load_AID1706_SARS_CoV_3CL())
```

### Case Study 3: 
Under the hood of one model from scratch, a flexible framework for method researchers:

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

# or simply load pretrained model from a model directory path or reproduced model name such as DeepDTA
net = models.model_pretrained(MODEL_PATH_DIR or MODEL_NAME)

# Repurpose using the trained model or pre-trained model
# In this example, loading repurposing dataset using Broad Repurposing Hub and SARS-CoV 3CL Protease Target.
X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
target, target_name = load_SARS_CoV_Protease_3CL()

_ = models.repurpose(X_repurpose, target, net, drug_name, target_name)
'''
Output:
------------------

...
'''

# Virtual screening using the trained model or pre-trained model 
# In this example, model is trained with binary outcome and customized input is given. 
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

_ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)
'''
Output:
------------------
Virtual Screening Result

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
We are currently in the testing release stage with active changing based on user feedback. After testing (few months), we will upload to conda for easier installation.

Checkout demos & tutorials in the [DEMO](https://github.com/kexinhuang12345/DeepPurpose/tree/master/DEMO) folder to start:

| Name | Description |
|-----------------|-------------|
| oneliner_repurpose_DEMO.ipynb | Repurposing for LCK target using oneline mode |
| DeepDTA_Reproduce.ipynb| Reproduce [DeepDTA](https://arxiv.org/abs/1801.10193) with DAVIS dataset|
| CNN-Binary-Example-DAVIS.ipynb| Binary Classification for DAVIS dataset using CNN encodings|


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



