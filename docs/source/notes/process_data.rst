Processing Data
======================================================================================================


Drug-Target Binding Benchmark Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We list public **Drug-Target Binding Benchmark Dataset** 
that is supported by DeepPurpose and 
corresponding downloading and processing function. 

===============   ===============================================================================================================
   Dataset                                        downloading and processing Function
===============   ===============================================================================================================
   BindingDB                         download_BindingDB() to download the data and process_BindingDB() to process the data
   DAVIS                             load_process_DAVIS() to download and process the data
   KIBA                               load_process_KIBA() to download and process the data
===============   ===============================================================================================================

* **Download Link**

	* `BindingDB <https://www.bindingdb.org/bind/index.jsp>`_ 
	* `DAVIS <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/>`_
	* `KIBA <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z>`_


Repurposing Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We list public **Repurposing Dataset** 
that is supported by DeepPurpose and 
corresponding downloading and processing function. 


=======================================   =====================================================================
           Dataset                                           downloading and processing Function   
=======================================   =====================================================================
     Curated Antiviral Drugs Library	          load_antiviral_drugs() to load and process the data  
       Broad Repurposing Hub	               load_broad_repurposing_hub() downloads and process the data  
=======================================   =====================================================================

* **Download Link**

	* `Curated Antiviral Drugs Library <https://en.wikipedia.org/wiki/List_of_antiviral_drugs>`_ 
	* `Broad Repurposing Hub <https://www.broadinstitute.org/drug-repurposing-hub>`_

Bioassay Data for COVID-19
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

=========================   =====================================================================
           Dataset                       downloading and processing Function   
=========================   =====================================================================
           AID1706		          load_AID1706_SARS_CoV_3CL() to load and process 
=========================   =====================================================================
* **Download Link**

	* `AID1706 <https://pubchem.ncbi.nlm.nih.gov/bioassay/1706>`_ 

COVID-19 Targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===============================   =====================================================================
           Dataset                       downloading and processing Function   
===============================   =====================================================================
  SARS-CoV 3CL Protease		          load_SARS_CoV_Protease_3CL()
  SARS-CoV2 3CL Protease	          load_SARS_CoV2_Protease_3CL()
  SARS_CoV2 RNA Polymerase	          load_SARS_CoV2_RNA_polymerase()
  SARS-CoV2 Helicase	              load_SARS_CoV2_Helicase()
  SARS-CoV2 3to5_exonuclease	      load_SARS_CoV2_3to5_exonuclease()
  SARS-CoV2 endoRNAse	               load_SARS_CoV2_endoRNAse()
===============================   =====================================================================



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Details of data processing function

   data/read_file_training_dataset_bioassay  
   data/read_file_training_dataset_drug_target_pairs
   data/read_file_virtual_screening_drug_target_pairs
   data/read_file_repurposing_library
   data/read_file_target_sequence
   data/download_BindingDB
   data/process_BindingDB
   data/load_process_DAVIS
   data/load_process_KIBA
























































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














