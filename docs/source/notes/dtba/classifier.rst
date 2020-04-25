Classifier
================================================






.. code-block:: python

	class DeepPurpose.models.Classifier(nn.Sequential) 



Classifier (`Source <https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/models.py#L318>`_) 
is to make the prediction for DBTA, it serve as a basic component of class DBTA. 


**constructor** create an instance of Classifier.

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





