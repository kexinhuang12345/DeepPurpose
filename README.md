<p align="center"><img src="figs/logo_deeppurpose_horizontal.png" alt="logo" width="400px" /></p>


<h3 align="center">
<p> A Drug Repurposing and Virtual Screening Toolkit with State-of-the-Art Deep Learning Methods</h3>

---

This repository hosts DeepPurpose, a Deep Learning Based Drug Repurposing and Virtual Screening Toolkit (using PyTorch). It allows extremely easy usage (only one line of code!) for any non-computational domain researchers to be able to obtain a list of potential drugs using state-of-the-art deep learning while facilitating deep learning method research in this field by providing a flexible framework (less than 10 lines of codes!) and baselines. 


### Features

- For non-computational researchers, ONE line of code from raw data to output drug repurposing/virtual screening result, aiming to allow wet-lab biochemists to leverage the power of deep learning. The result is ensembled from five pretrained deep learning models!

- For computational researchers, 15+ powerful encodings for drugs and proteins, ranging from deep neural network on classic cheminformatics fingerprints, CNN, transformers to message passing graph neural network, with 50+ combined models! Most of the combinations of the encodings are not yet in existing works. All of these under 10 lines but with lots of flexibility! Switching encoding is as simple as changing the encoding names!

- Realistic and user-friendly design: 
	- automatic identification to do drug target binding affinity (regression) or drug target interaction prediction (binary) task.
	- support cold target, cold drug settings for robust model evaluations and support single-target high throughput sequencing assay data setup.
	- many dataset loading/downloading/unzipping scripts to ease the tedious preprocessing, including antiviral, COVID19 targets, BindingDB, DAVIS, KIBA, ...
	- many pretrained checkpoints.
	- easy monitoring of training process with detailed training metrics output such as test set figures (AUCs) and tables, also support early stopping.
	- detailed output records such as rank list for repurposing result.
	- various evaluation metrics: ROC-AUC, PR-AUC, F1 for binary task, MSE, R-squared, Concordance Index for regression task.
	- label unit conversion for skewed label distribution such as Kd.
	- time reference for computational expensive encoding.
	- PyTorch based, support CPU, GPU, Multi-GPUs.
	
*NOTE: We are actively looking for constructive advices/user feedbacks/experiences on using DeepPurpose! Please open an issue or [contact us](kexinhuang@hsph.harvard.edu).*
	

## Example

### Case Study 1 (a): Antiviral Drugs Repurposing for SARS-CoV2 3CLPro, using One Line.
Given a new target sequence (e.g. SARS-CoV2 3CL Protease), retrieve a list of repurposing drugs from a curated drug library of 81 antiviral drugs. The Binding Score is the Kd values. Results aggregated from five pretrained model on BindingDB dataset!

```python
from DeepPurpose import oneliner
oneliner.repurpose(*load_SARS_CoV2_Protease_3CL(), *load_antiviral_drugs())
```
```
----output----
Drug Repurposing Result for SARS-CoV2 3CL Protease
+------+----------------------+------------------------+---------------+
| Rank |      Drug Name       |      Target Name       | Binding Score |
+------+----------------------+------------------------+---------------+
|  1   |      Sofosbuvir      | SARS-CoV2 3CL Protease |     190.25    |
|  2   |     Daclatasvir      | SARS-CoV2 3CL Protease |     214.58    |
|  3   |      Vicriviroc      | SARS-CoV2 3CL Protease |     315.70    |
|  4   |      Simeprevir      | SARS-CoV2 3CL Protease |     396.53    |
|  5   |      Etravirine      | SARS-CoV2 3CL Protease |     409.34    |
|  6   |      Amantadine      | SARS-CoV2 3CL Protease |     419.76    |
|  7   |      Letermovir      | SARS-CoV2 3CL Protease |     460.28    |
|  8   |     Rilpivirine      | SARS-CoV2 3CL Protease |     470.79    |
|  9   |      Darunavir       | SARS-CoV2 3CL Protease |     472.24    |
|  10  |      Lopinavir       | SARS-CoV2 3CL Protease |     473.01    |
|  11  |      Maraviroc       | SARS-CoV2 3CL Protease |     474.86    |
|  12  |    Fosamprenavir     | SARS-CoV2 3CL Protease |     487.45    |
|  13  |      Ritonavir       | SARS-CoV2 3CL Protease |     492.19    |
....
```


### Case Study 1 (b): New Target Repurposing using Broad Drug Repurposing Hub, with One Line.
Given a new target sequence (e.g. MMP9), retrieve a list of repurposing drugs from Broad Drug Repurposing Hub, which is the default. Results also aggregated from five pretrained model! Note the drug name here is the Pubchem CID since some drug names in Broad is too long.

```python
from DeepPurpose import oneliner
oneliner.repurpose(*load_MMP9())
```
```
----output----
Drug Repurposing Result for MMP9
+------+-------------+-------------+---------------+
| Rank |  Drug Name  | Target Name | Binding Score |
+------+-------------+-------------+---------------+
|  1   |  6917849.0  |     MMP9    |      5.42     |
|  2   |   441336.0  |     MMP9    |      6.97     |
|  3   |   441335.0  |     MMP9    |      8.37     |
|  4   |   27924.0   |     MMP9    |      9.84     |
|  5   |   16490.0   |     MMP9    |      9.86     |
|  6   |  23947600.0 |     MMP9    |     10.11     |
|  7   |    5743.0   |     MMP9    |     12.44     |
|  8   |    3288.0   |     MMP9    |     15.91     |
|  9   | 129009989.0 |     MMP9    |     18.01     |
|  10  | 129009925.0 |     MMP9    |     23.13     |
|  11  |  40467076.0 |     MMP9    |     23.48     |
|  12  |  6917974.0  |     MMP9    |     24.50     |
|  13  |  73707512.0 |     MMP9    |     26.83     |
```

### Case Study 2: Repurposing using Customized training data, with One Line.
Given a new target sequence (e.g. SARS-CoV 3CL Pro), training on new data (AID1706 Bioassay), and then retrieve a list of repurposing drugs from a proprietary library (e.g. antiviral drugs). The model can be trained from scratch or finetuned from the pretraining checkpoint!

```python
from DeepPurpose import oneliner
from DeepPurpose.dataset import *

oneliner.repurpose(*load_SARS_CoV_Protease_3CL(), *load_antiviral_drugs(no_cid = True),  *load_AID1706_SARS_CoV_3CL(), \
		split='HTS', convert_y = False, frac=[0.8,0.1,0.1], pretrained = False, agg = 'max_effect')
```
```
----output----
Drug Repurposing Result for SARS-CoV 3CL Protease
+------+----------------------+-----------------------+-------------+-------------+
| Rank |      Drug Name       |      Target Name      | Interaction | Probability |
+------+----------------------+-----------------------+-------------+-------------+
|  1   |      Remdesivir      | SARS-CoV 3CL Protease |     YES     |     0.99    |
|  2   |      Efavirenz       | SARS-CoV 3CL Protease |     YES     |     0.98    |
|  3   |      Vicriviroc      | SARS-CoV 3CL Protease |     YES     |     0.98    |
|  4   |      Tipranavir      | SARS-CoV 3CL Protease |     YES     |     0.96    |
|  5   |     Methisazone      | SARS-CoV 3CL Protease |     YES     |     0.94    |
|  6   |      Letermovir      | SARS-CoV 3CL Protease |     YES     |     0.88    |
|  7   |     Idoxuridine      | SARS-CoV 3CL Protease |     YES     |     0.77    |
|  8   |       Loviride       | SARS-CoV 3CL Protease |     YES     |     0.76    |
|  9   |      Baloxavir       | SARS-CoV 3CL Protease |     YES     |     0.74    |
|  10  |     Ibacitabine      | SARS-CoV 3CL Protease |     YES     |     0.70    |
|  11  |     Taribavirin      | SARS-CoV 3CL Protease |     YES     |     0.65    |
|  12  |      Indinavir       | SARS-CoV 3CL Protease |     YES     |     0.62    |
|  13  |   Podophyllotoxin    | SARS-CoV 3CL Protease |     YES     |     0.60    |
....
```

### Case Study 3: A Framework for Drug Target Interaction Prediction, with less than 10 lines of codes.
Under the hood of one model from scratch, a flexible framework for method researchers:

```python
from DeepPurpose import models
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
config = generate_config(drug_encoding, target_encoding, transformer_n_layer_target = 8)
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

# Virtual screening using the trained model or pre-trained model 
# In this example, model is trained with binary outcome and customized input is given. 
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

_ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)

```

## Install & Usage
Try it on [Binder](https://mybinder.org)! Binder is a cloud Jupyter Notebook interface that will install our environment dependency for you. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kexinhuang12345/DeepPurpose/master)

We recommend to install it locally since Binder needs to install environment every time launching:

```bash
# -- First Time -- #
git clone git@github.com:kexinhuang12345/DeepPurpose.git
# Download code repository

cd DeepPurpose
# Change directory to DeepPurpose

conda env create -f env.yml  
## Build virtual environment with all packages installed using conda

conda activate DeepPurpose
## Activate conda environment

jupyter notebook
## open the jupyter notebook with the conda env

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

Docker image will also be up soon!

We also provide installation video tutorials:

\[[Binder]()\]

\[[Local PC conda]()\]

*We are currently in the testing release stage with frequent modifications based on user feedback. After testing (few months), we will upload to conda for release, which could have easier installation.*

## Demos
Checkout 10+ demos & tutorials in the [DEMO](https://github.com/kexinhuang12345/DeepPurpose/tree/master/DEMO) folder to start:

| Name | Description |
|-----------------|-------------|
| [Drug Repurposing for 3CLPro](DEMO/case-study-I-Drug-Repurposing-for-3CLPro.ipynb)| Example of one-liner repurposing for 3CLPro|
| [Drug Repurposing with Customized Data](DEMO/case-study-III-Drug-Repurposing-with-Customized-Data.ipynb)| Example of one-liner repurposing with AID1706 Bioassay Data, training from scratch|
| [Virtual Screening for BindingDB IC50](DEMO/case-study-II-Virtual-Screening-for-BindingDB-IC50.ipynb) | Example of one-liner virtual screening |
|[Reproduce_DeepDTA](DEMO/case-study-IV-Reproduce_DeepDTA.ipynb)|Reproduce [DeepDTA](https://arxiv.org/abs/1801.10193) with DAVIS dataset and show how to use the 10 lines framework|
| [Binary Classification for DAVIS using CNNs](DEMO/CNN-Binary-Example-DAVIS.ipynb)| Binary Classification for DAVIS dataset using CNN encodings by using the 10 lines framework.|
| [Dataset Tutorial](DEMO/load_data_tutorial.ipynb) | Tutorial on how to use the dataset loader and read customized data|
| [Pretraining Model Tutorial](DEMO/load_pretraining_models_tutorial.ipynb)| Tutorial on how to load pretraining models|

....

## Cite Us

Please cite [arxiv]():
```
@article{
}

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

## Data
DeePurpose supports the following dataset loaders for now and more will be added:

*Public Drug-Target Binding Benchmark Dataset*
| Data  | Function |
|-------|----------|
|[BindingDB](https://www.bindingdb.org/bind/index.jsp)| ```download_BindingDB()``` to download the data and ```process_BindingDB()* ``` to process the data|
|[DAVIS](http://staff.cs.utu.fi/~aatapa/data/DrugTarget/)|```load_process_DAVIS() ``` to download and process the data|
|[KIBA](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)|```load_process_KIBA() ``` to download and process the data|

*Repurposing Dataset*
| Data  | Function |
|-------|----------|
|[Curated Antiviral Drugs Library](https://en.wikipedia.org/wiki/List_of_antiviral_drugs)|```load_antiviral_drugs()``` to load and process the data|
|[Broad Repurposing Hub](https://www.broadinstitute.org/drug-repurposing-hub)|```load_broad_repurposing_hub() ``` downloads and process the data|

*Bioassay Data for COVID-19*
(Thanks to [MIT AI Cures](https://www.aicures.mit.edu/data))
| Data  | Function |
|-------|----------|
|[AID1706](https://pubchem.ncbi.nlm.nih.gov/bioassay/1706)|```load_AID1706_SARS_CoV_3CL()``` to load and process|

*COVID-19 Targets*
| Data  | Function |
|-------|----------|
|SARS-CoV 3CL Protease|```load_SARS_CoV_Protease_3CL()```|
|SARS-CoV2 3CL Protease|```load_SARS_CoV2_Protease_3CL()```|
|SARS_CoV2 RNA Polymerase|```load_SARS_CoV2_RNA_polymerase()```|
|SARS-CoV2 Helicase|```load_SARS_CoV2_Helicase()```|
|SARS-CoV2 3to5_exonuclease|```load_SARS_CoV2_3to5_exonuclease()```|
|SARS-CoV2 endoRNAse|```load_SARS_CoV2_endoRNAse()```|

DeepPurpose also supports to read from users' txt file. It assumes the following data format.

For drug target pairs:
```
Drug1_SMILES Target1_Seq Score/Label
Drug2_SMILES Target2_Seq Score/Label
....
```
Then, use 

```python 
from DeepPurpose import dataset
X_drug, X_target, y = dataset.read_file_training_dataset_drug_target_pairs(PATH)
```

For bioassay training data:
```
Target_Seq
Drug1_SMILES Score/Label
Drug2_SMILES Score/Label
....
```

Then, use 

```python 
from DeepPurpose import dataset
X_drug, X_target, y = dataset.read_file_training_dataset_bioassay(PATH)
```

For drug repurposing library:
```
Drug1_Name Drug1_SMILES 
Drug2_Name Drug2_SMILES
....
```
Then, use 

```python 
from DeepPurpose import dataset
X_drug, X_drug_names = dataset.read_file_repurposing_library(PATH)
```

For target sequence to be repurposed:
```
Target_Name Target_seq 
```
Then, use 

```python 
from DeepPurpose import dataset
Target_seq, Target_name = dataset.read_file_target_sequence(PATH)
```

For virtual screening library:
```
Drug1_SMILES Drug1_Name Target1_Seq Target1_Name
Drug1_SMILES Drug1_Name Target1_Seq Target1_Name
....
```
Then, use 

```python 
from DeepPurpose import dataset
X_drug, X_target, X_drug_names, X_target_names = dataset.read_file_virtual_screening_drug_target_pairs(PATH)
```
Checkout [Dataset Tutorial](DEMO/load_data_tutorial.ipynb).

## Pretrained models
We provide more than 10 pretrained models. Please see [Pretraining Model Tutorial](DEMO/load_pretraining_models_tutorial.ipynb) on how to load them. It is as simple as 

```python
from DeepPurpose import models
net = models.model_pretrained(model = 'MPNN_CNN_DAVIS')
or
net = models.model_pretrained(FILE_PATH)
```
The list of avaiable pretrained models:

Model name consists of first the drug encoding, then the target encoding and then the trained dataset.

|Model Name||
|------|
|DeepDTA_DAVIS|
|CNN_CNN_BindingDB|
|Morgan_CNN_BindingDB|
|Morgan_CNN_KIBA|
|Morgan_CNN_DAVIS|
|MPNN_CNN_BindingDB|
|MPNN_CNN_KIBA|
|MPNN_CNN_DAVIS|
|Transformer_CNN_BindingDB|
|Daylight_AAC_DAVIS|
|Daylight_AAC_KIBA|
|Daylight_AAC_BindingDB|
|Morgan_AAC_BindingDB|
|Morgan_AAC_KIBA|
|Morgan_AAC_DAVIS|
|CNN_Transformer_DAVIS|

## Documentations
More detailed documentations are coming soon.

## Contact
Please contact kexinhuang@hsph.harvard.edu or tfu42@gatech.edu for help or submit an issue. 

## Disclaimer
The output list should be inspected manually by experts before proceeding to the wet-lab validation, and our work is still in active developement with limitations, please do not directly use the drugs.


