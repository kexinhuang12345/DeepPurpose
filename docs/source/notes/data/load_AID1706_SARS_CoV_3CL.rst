load_AID1706_SARS_CoV_3CL
========================================================================================================


**load_AID1706_SARS_CoV_3CL** load AID1706_SARS_CoV_3CL dataset. 

.. code-block:: python

  load_AID1706_SARS_CoV_3CL(path = './data', binary = True, threshold = 15, balanced = True, oversample_num = 30, seed = 1)

* **path** (str, a directory) - the path that save AID1706_SARS_CoV_3CL dataset file. Example: "./data". 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **threshold** (float) - The threshold that select target score ?? 
* **balanced** (bool) - If True, do oversampling to make number of positive and negative samples equal. 
* **oversample_num** (int) - control the oversample rate. 
* **seed** (int) - random seed in oversample. 







