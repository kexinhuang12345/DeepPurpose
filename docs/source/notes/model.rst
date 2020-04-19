Model
========================




.. code-block:: python

	class DeepPurpose.models.Classifier(nn.Sequential)

Classifier is make the prediction for DBTA, it serve as a basic component of class DBTA. 


**constructor**

.. code-block:: python

	__init__(self, model_drug, model_protein, **config) 


* **model_drug** (DeepPurpose.models.XX) - Encoder model for drug. XX can be "transformer", "MPNN", "CNN", "CNN_RNN" ..., 
* **model_protein** (DeepPurpose.models.XX) - Encoder model for protein. XX can be "transformer", "CNN", "CNN_RNN" ..., 
* **config** (kwargs, keyword arguments) - specify the parameter of classifier.  



**Calling functions** implement the feedforward procedure of Classifier.


.. code-block:: python

	forward(self, v_D, v_P)


* **v_D** (many types) - input feature for drug encoder model, like "DeepPurpose.models.transformer", "DeepPurpose.models.CNN", "DeepPurpose.models.CNN_RNN", "DeepPurpose.models.MPNN". 
* **v_P** (many types) - input feature for protein encoder model, like "DeepPurpose.models.transformer", "DeepPurpose.models.CNN", "DeepPurpose.models.CNN_RNN".  




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

	class DeepPurpose.models.DBTA

Drug Target Binding Affinity (DBTA) model include all learning procedure. 


name of function: **constructor** create DBTA model. 

.. code-block:: python

	__init__(self, **config)


* **config** (kwargs, keyword arguments) - specify the parameter of DBTA.  
	* **drug_encoding** (str) - Encoder mode for drug. It can be "transformer", "MPNN", "CNN", "CNN_RNN" ...,
	* **target_encoding** (str) - Encoder mode for protein. It can be "transformer", "CNN", "CNN_RNN" ..., 
	* **result_folder** (str) - directory that store the learning log/results. 
	* **concrete parameter for encoder model** (repeated)


name of function: **test_** include all the inference procedure. 

.. code-block:: python


	test_(self, data_generator, model, repurposing_mode = False, test = False):

* **data_generator** (iterator) - iterator of torch.utils.data.DataLoader. It can be test data or validation data. 
* **model** (DeepPurpose.models.Classifier) - model of DBTA. 
* **repurposing_mode** (bool) - If repurposing_mode is True, then do repurposing. Otherwise, do compute the accuracy (including AUC score). 
* **test** (bool) - If test is True, plot ROC-AUC and PR-AUC curve. Otherwise, pass. 


name of function: **train** include all the training procedure. 

.. code-block:: python

	train(self, train, val, test = None, verbose = True)

* **train** () - Train data generator
* **val** () - Valid data generator
* **test** () - Test data generator
* **verbose** (bool) - If verbose is True, then print training record every 100 iterations. 


name of function: **predict** 

.. code-block:: python

	 predict(self, df_data)

* **df_data** (pd.DataFrame) - specify data that we need to predict. 


name of function: **save_model** save the well-trained model to specific directory. 

.. code-block:: python

	save_model(self, path_dir) 

* **path_dir** (str, a directory) - the path where model is saved. 



name of function: **load_pretrained** load the well-trained model so that we are able to make inference directly. 

.. code-block:: python

	load_pretrained(self, path)

* **path** (str, a directory) - the path where model is loaded. 





































