load bioarray dataset (read_file_training_dataset_bioassay)
========================================================================================================





**read_file_repurposing_library** load drug repurposing dataset. 
We have requirement on format of file. 
Each line contains a drug SMILES and its name.  
Example: ./toy_data/??


.. code-block:: python

	dataset.read_file_repurposing_library(path)

* **path** (str, a directory) - the path of drug repurposing dataset file. 




