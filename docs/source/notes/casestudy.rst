Case Study  
================================================

* **1a. Antiviral Drugs Repurposing for SARS-CoV2 3CLPro, using One Line.**

Given a new target sequence (e.g. SARS-CoV2 3CL Protease), 
retrieve a list of repurposing drugs from a curated drug library of 81 antiviral drugs. 
The Binding Score is the Kd values. 
Results aggregated from five pretrained model on BindingDB dataset!

.. code-block:: python


	from DeepPurpose import oneliner
	oneliner.repurpose(*load_SARS_CoV2_Protease_3CL(), *load_antiviral_drugs())




* **1b. New Target Repurposing using Broad Drug Repurposing Hub, with One Line.**


Given a new target sequence (e.g. MMP9), 
retrieve a list of repurposing drugs from Broad Drug Repurposing Hub, 
which is the default. 
Results also aggregated from five pretrained model! 
Note the drug name here is the Pubchem CID since some drug names in Broad is too long.

.. code-block:: python

	from DeepPurpose import oneliner
	oneliner.repurpose(*load_MMP9())




* **2. Repurposing using Customized training data, with One Line.**


Given a new target sequence (e.g. SARS-CoV 3CL Pro), 
training on new data (AID1706 Bioassay), 
and then retrieve a list of repurposing drugs from a proprietary library (e.g. antiviral drugs). 
The model can be trained from scratch or finetuned from the pretraining checkpoint!



.. code-block:: python


	from DeepPurpose import oneliner
	from DeepPurpose.dataset import *

	oneliner.repurpose(*load_SARS_CoV_Protease_3CL(), *load_antiviral_drugs(no_cid = True),  *load_AID1706_SARS_CoV_3CL(), \
		split='HTS', convert_y = False, frac=[0.8,0.1,0.1], pretrained = False, agg = 'max_effect')









* 3. **A Framework for Drug Target Interaction Prediction, with less than 10 lines of codes.**

Under the hood of one model from scratch, a flexible framework for method researchers:

.. code-block:: python


	from DeepPurpose import models
	from DeepPurpose.utils import *
	from DeepPurpose.dataset import *

	# Load Data, an array of SMILES for drug,
	# an array of Amino Acid Sequence for Target 
	# and an array of binding values/0-1 label.
	# e.g. ['Cc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1', ...], 
	#      ['MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTH...', ...], 
	#      [0.46, 0.49, ...]
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

	# Generate new model using default parameters; 
	# also allow model tuning via input parameters.
	config = generate_config(drug_encoding, target_encoding, \
							 transformer_n_layer_target = 8)
	net = models.model_initialize(**config)

	# Train the new model.
	# Detailed output including a tidy table storing 
	#    validation loss, metrics, AUC curves figures and etc. 
	#    are stored in the ./result folder.
	net.train(train, val, test)

	# or simply load pretrained model from a model directory path 
	#   or reproduced model name such as DeepDTA
	net = models.model_pretrained(MODEL_PATH_DIR or MODEL_NAME)

	# Repurpose using the trained model or pre-trained model
	# In this example, loading repurposing dataset using 
	# Broad Repurposing Hub and SARS-CoV 3CL Protease Target.
	X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
	target, target_name = load_SARS_CoV_Protease_3CL()

	_ = models.repurpose(X_repurpose, target, net, drug_name, target_name)

	# Virtual screening using the trained model or pre-trained model 
	X_repurpose, drug_name, target, target_name = \
			['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], \
			['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...],\
			['P36896', 'P00374']

	_ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)






* 4. **Virtual Screening with Customized Training Data with One Line**

Given a list of new drug-target pairs to be screened, 
retrieve a list of drug-target pairs with top predicted binding scores. 


.. code-block:: python

	from DeepPurpose import oneliner
	oneliner.virtual_screening(['MKK...LIDL', ...], ['CC1=C...C4)N', ...])








