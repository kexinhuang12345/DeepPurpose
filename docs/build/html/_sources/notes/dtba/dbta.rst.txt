Drug Target Binding Affinity (DTBA) Model
================================================


.. code-block:: python

	class DeepPurpose.models.DBTA

**Drug Target Binding Affinity (DBTA)** (`Source <https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/models.py#L509>`_)  include all component, including drug encoder, target encoder and classifier/regressor. 


**constructor** create  an instance of DBTA. 

.. code-block:: python

	__init__(self, **config)


* **config** (kwargs, keyword arguments) - specify the parameter of DBTA.  
	* **drug_encoding** (str) - Encoder mode for drug. It can be "transformer", "MPNN", "CNN", "CNN_RNN" ...,
	* **target_encoding** (str) - Encoder mode for protein. It can be "transformer", "CNN", "CNN_RNN" ..., 
	* **result_folder** (str) - directory that store the learning log/results. 
	* **concrete parameter for encoder model** (repeated)


**test_** include all the test procedure. 

.. code-block:: python


	test_(self, data_generator, model, repurposing_mode = False, test = False):

* **data_generator** (iterator) - iterator of torch.utils.data.DataLoader. It can be test data or validation data. 
* **model** (DeepPurpose.models.Classifier) - model of DBTA. 
* **repurposing_mode** (bool) - If repurposing_mode is True, then do repurposing. Otherwise, do compute the accuracy (including AUC score). 
* **test** (bool) - If test is True, plot ROC-AUC and PR-AUC curve. Otherwise, pass. 


**train** include all the training procedure. 

.. code-block:: python

	train(self, train, val, test = None, verbose = True)

* **train** (torch.utils.data.dataloader) - Train data loader
* **val** (torch.utils.data.dataloader) - Valid data loader
* **test** (torch.utils.data.dataloader) - Test data loader
* **verbose** (bool) - If verbose is True, then print training record every 100 iterations. 


**predict** include all the inference procedure. 

.. code-block:: python

	 predict(self, df_data)

* **df_data** (pd.DataFrame) - specify data that we need to predict. 


**save_model** save the well-trained model to specific directory. 

.. code-block:: python

	save_model(self, path_dir) 

* **path_dir** (str, a directory) - the path where model is saved. 



**load_pretrained** load the well-trained model so that we are able to make inference directly and don't have to train model from scratch. 

.. code-block:: python

	load_pretrained(self, path)

* **path** (str, a directory) - the path where model is loaded. 






