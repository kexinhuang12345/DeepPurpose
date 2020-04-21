read_file_virtual_screening_drug_target_pairs
========================================================================================================




**read_file_virtual_screening_drug_target_pairs** load virtual screening drug target pairs dataset. 
We have requirement on format of file. 
Each line contains a drug SMILES and target sequence. 
Example: ./toy_data/dti.txt 

.. code-block:: python

	dataset.read_file_virtual_screening_drug_target_pairs(path)


* **path** (str, a directory) - the path of virtual screening drug target pairs dataset file. 






