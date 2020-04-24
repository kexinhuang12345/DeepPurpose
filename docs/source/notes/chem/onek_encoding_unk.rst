DeepPurpose.chemutils.onek_encoding_unk
================================================


Given an atom and an allowable atom set, 
allowable atom set contains a special symbol for unknown atom. 
The target of onek_encoding_unk function is to 
transform the atom into one-hot vector. 
If the atom doesn't exist in the allowable atom set, the use label it as unknown atom. 




.. code-block:: python

	def onek_encoding_unk(x, allowable_set):
	    if x not in allowable_set:
	        x = allowable_set[-1]
	    return list(map(lambda s: x == s, allowable_set))






