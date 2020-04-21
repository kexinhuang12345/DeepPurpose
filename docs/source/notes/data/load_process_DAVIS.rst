load_process_DAVIS
========================================================================================================










**load_process_DAVIS**  load DAVIS dataset. 

.. code-block:: python

	dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)


* **path** (str, a directory) - the path that save DAVIS dataset file. Example: "./data". 
* **binary** (bool) - If binary is True, formulate prediction task as a binary classification task. Otherwise, formulate the prediction task as a regression task. 
* **convert_to_log** (bool) - If True, convert the target score to logspace for easier regression'
* **threshold** (float) - The threshold that select target score ?? 




