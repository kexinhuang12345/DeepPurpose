read_file_training_dataset_bioassay
========================================================================================================

**read_file_training_dataset_bioassay** load bioarray dataset, with one target sequence and multiple drugs and their interaction score with the target. 

.. code-block:: python

	dataset.read_file_training_dataset_bioassay(path)


* **path** (str, a directory) - the path of bioassay dataset file. We have requirement on format of file. First line is target sequence. From 2nd line to n-th line, each line a SMILES and interaction score with target sequence. Example: ./toy_data/AID1706.txt 



