<h1 align="center">
<p>DeepPurpose
</h1>

<h3 align="center">
<p> A Drug Repurposing and Virtual Screening Toolkit with \n State-of-the-Art Deep Learning Methods
</h3>

This repository hosts DeepPurpose, a Deep Learning Based Drug Repurposing and Virtual Screening Toolkit. It allows extremely easy usage (only ten lines of codes) for any non-computational domain researchers to be able to obtain a list of potential drugs using state-of-the-art deep learning. 


### Features

- 10 lines of code from raw data to output drug repurposing/virtual screening result, designed to allow wet-lab biochemists to leverage the power of deep learning and machine learning researchers to push forward the frontiers of DTI prediction.

- 15+ state-of-the-art encodings for drugs and proteins, ranging from deep neural network on classic cheminformatics fingerprints, CNN-RNN, transformers to message passing graph neural network. Switching encoding is as simple as changing the encoding names!

- Realistic and user-friendly design: 
	- automatic identification to do drug target binding affinity (regression) or drug target interaction prediction (binary) task.
	- support cold target, cold drug settings for robust model evaluations and support one-target high throughput sequencing assay data setup.
	- label unit conversion for skewed label distribution such as Kd.
	- time reference for computational expensive encoding.
	- easy monitoring of training process with early stopping on validation set. 
	- various evaluation metrics: ROC-AUC, PR-AUC, F1 for binary task, MSE, R-squared, Concordance Index for regression task.
	- allow easy pretraining models loading 

## Example

```python
import DeepPurpose.models as models
from DeepPurpose.utils import *

# Load Data, SMILES for drug, Amino Acid Sequence for Target. 
# Support both continuous (IC50, Kd, etc.) and binary (interaction or not). Automatically adjust the loss and evaluation metrics.
X_drug, X_target, y  = ['Cc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1', ...], ['MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTH...', ...], [0.46, 0.49, ...]

# Type in the encoding names for drug/protein.
drug_encoding, target_encoding = 'MPNN', 'Transformer'

# Data processing, allow cold drug/protein setup split.
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='cold_protein', 
                                frac=[0.7,0.1,0.2])

# Generate model, allow easy model tuning via configs
config = generate_config(drug_encoding, target_encoding, transformer_n_layer_target = 3)
model = models.model_initialize(**config)

# Train the model and support early stopping
model.train(train, val, test)

# Repurpose using the trained model or pre-trained model
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], 'MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', 'P36507'

_ = models.repurpose(X_repurpose, target, model, drug_name, target_name)
'''
Output:
------------------
Drug Repurposing Result for P36507
Drug 16007391   predicted to have binding affinity score 4.87
Drug 447077     predicted to have binding affinity score 5.39
Drug 24860417   predicted to have binding affinity score 4.70
...
'''

# Virtual screening using the trained model or pre-trained model (in this example, model is trained with binary outcome)
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

_ = models.virtual_screening(X_repurpose, target, model, drug_name, target_name)
'''
Output:
------------------
Virtual Screening Result
Drug 16007391   predicted to NOT have interaction with the target P36896 
Drug 44355753   predicted to have interaction with the target P00374 
Drug 24180719   predicted to NOT have interaction with the target P61075 
'''
```

## Install
```
pip install deeppurpose
```

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
Please contact kexinhuang@hsph.harvard.edu for help or submit an issue. 



