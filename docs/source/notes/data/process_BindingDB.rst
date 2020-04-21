process_BindingDB
========================================================================================================






**process_BindingDB**  processes BindingDB dataset. 

.. code-block:: python

	dataset.process_BindingDB(path = None, df = None, y = 'Kd', binary = False, convert_to_log = True, threshold = 30)

* **path** (str, a directory) - the path that save BindingDB dataset file. Example: "./data/BindingDB_All.tsv". 
* **df** (pandas.DataFrame) - Dataframe that contains input data, if first parameter "path" is None, use the "df". 
* **y** (str; can be "Kd", "Ki", "IC50" or "EC50") - specify the binding score. 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **convert_to_log** (bool) - If True, convert the target score to logspace for easier regression'
* **threshold** (float) - The threshold that select target score ?? 








