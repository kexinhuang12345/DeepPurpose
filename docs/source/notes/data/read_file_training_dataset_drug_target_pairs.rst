read_file_training_dataset_drug_target_pairs
========================================================================================================




**read_file_training_dataset_drug_target_pairs** load drug target pairs dataset. We have requirement on format of file. Each line contains a drug SMILES and target sequence and their interaction score. Example: ./toy_data/dti.txt 


.. code-block:: python

	dataset.read_file_training_dataset_drug_target_pairs(path)

* **path** (str, a directory) - the path of drug target pairs dataset file. We have requirement on format of file. First line is target sequence. From 2nd line to n-th line, each line a SMILES and interaction score with target sequence.  Example: ./toy_data/AID1706.txt 






