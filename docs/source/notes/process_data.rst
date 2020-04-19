Processing Data
========================



name of function: **read_file_training_dataset_bioassay** load bioarray dataset, with one target sequence and multiple drugs and their interaction score with the target. 

.. code-block:: python

	dataset.read_file_training_dataset_bioassay(path)


* **path** (str, a directory) - the path of bioassay dataset file. We have requirement on format of file. First line is target sequence. From 2nd line to n-th line, each line a SMILES and interaction score with target sequence. Example: ./toy_data/AID1706.txt 




name of function: **read_file_training_dataset_drug_target_pairs** load drug target pairs dataset. We have requirement on format of file. Each line contains a drug SMILES and target sequence and their interaction score. Example: ./toy_data/dti.txt 


.. code-block:: python

	dataset.read_file_training_dataset_drug_target_pairs(path)

* **path** (str, a directory) - the path of drug target pairs dataset file. We have requirement on format of file. First line is target sequence. From 2nd line to n-th line, each line a SMILES and interaction score with target sequence.  Example: ./toy_data/AID1706.txt 






name of function: **read_file_virtual_screening_drug_target_pairs** load virtual screening drug target pairs dataset. 
We have requirement on format of file. 
Each line contains a drug SMILES and target sequence. 
Example: ./toy_data/dti.txt 

.. code-block:: python

	dataset.read_file_virtual_screening_drug_target_pairs(path)


* **path** (str, a directory) - the path of virtual screening drug target pairs dataset file. 









name of function: **read_file_repurposing_library** load drug repurposing dataset. 
We have requirement on format of file. 
Each line contains a drug SMILES and its name.  
Example: ./toy_data/??


.. code-block:: python

	dataset.read_file_repurposing_library(path)

* **path** (str, a directory) - the path of drug repurposing dataset file. 








name of function: **read_file_target_sequence** load drug repurposing dataset. 
We have requirement on format of file.
The file only have one line. 
The line contains target name and target sequence. 
Example: ./toy_data/??

.. code-block:: python

	dataset.read_file_target_sequence(path)

* **path** (str, a directory) - the path of target sequence dataset file. 











name of function: **download_BindingDB** load BindingDB dataset, save it to a specific path. 
If the path doesn't exist, create the folder. 

.. code-block:: python

	dataset.download_BindingDB(path)

* **path** (str, a directory) - the path that save BindingDB dataset file. Example: "./data". 












name of function: **download_DrugTargetCommons** load DrugTargetCommons dataset, save it to a specific path. 
If the path doesn't exist, create the folder. 

.. code-block:: python

	dataset.download_DrugTargetCommons(path)

* **path** (str, a directory) - the path that save DrugTargetCommons dataset file. Example: "./data". 













name of function: **process_BindingDB**  processes BindingDB dataset. 

.. code-block:: python

	dataset.process_BindingDB(path = None, df = None, y = 'Kd', binary = False, convert_to_log = True, threshold = 30)

* **path** (str, a directory) - the path that save BindingDB dataset file. Example: "./data/BindingDB_All.tsv". 
* **df** (pandas.DataFrame) - Dataframe that contains input data, if first parameter "path" is None, use the "df". 
* **y** (str; can be "Kd", "Ki", "IC50" or "EC50") - specify the binding score. 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **convert_to_log** (bool) - If True, convert the target score to logspace for easier regression'
* **threshold** (float) - The threshold that select target score ?? 






















name of function: **load_process_DAVIS**  load DAVIS dataset. 

.. code-block:: python

	dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)


* **path** (str, a directory) - the path that save DAVIS dataset file. Example: "./data". 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **convert_to_log** (bool) - If True, convert the target score to logspace for easier regression'
* **threshold** (float) - The threshold that select target score ?? 

















name of function: **load_process_KIBA** load KIBA dataset. 


.. code-block:: python


	load_process_KIBA(path = './data', binary = False, threshold = 9):

* **path** (str, a directory) - the path that save KIBA dataset file. Example: "./data". 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **threshold** (float) - The threshold that select target score ?? 










name of function: **load_AID1706_txt_file** load KIBA dataset. 

.. code-block:: python


	load_AID1706_txt_file(path = './data')

* **path** (str, a directory) - the path that save AID1706 dataset file. Example: "./data". 

















name of function: **load_AID1706_SARS_CoV_3CL** load AID1706_SARS_CoV_3CL dataset. 

.. code-block:: python

	load_AID1706_SARS_CoV_3CL(path = './data', binary = True, threshold = 15, balanced = True, oversample_num = 30, seed = 1)

* **path** (str, a directory) - the path that save AID1706_SARS_CoV_3CL dataset file. Example: "./data". 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **threshold** (float) - The threshold that select target score ?? 
* **balanced** (bool) - If True, do oversampling to make number of positive and negative samples equal. 
* **oversample_num** (int) - control the oversample rate. 
* **seed** (int) - random seed in oversample. 












name of function: **load_broad_repurposing_hub** load repurposing dataset. 

.. code-block:: python

	load_broad_repurposing_hub(path = './data'):

* **path** (str, a directory) - the path that save repurposing dataset file. Example: "./data". 















name of function: **load_antiviral_drugs** load antiviral drugs dataset. 

.. code-block:: python

	load_antiviral_drugs(path = './data', no_cid = False)

* **path** (str, a directory) - the path that save antiviral drugs dataset file. Example: "./data". 
* **no_cid** (bool) - If False, including "Pubchem CID". 














