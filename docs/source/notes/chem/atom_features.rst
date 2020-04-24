DeepPurpose.chemutils.atom_features
================================================


Given an atom in molecular graph, return its feature based on the atom itself, its degree and other information. 

.. code-block:: python

	def atom_features(atom):
	    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
	            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
	            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
	            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
	            + [atom.GetIsAromatic()])



