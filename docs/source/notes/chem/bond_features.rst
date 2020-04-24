DeepPurpose.chemutils.bond_features
================================================


Given a bond in molecular graph, return its feature based on the bond itself, its connection information. 

.. code-block:: python

	def bond_features(bond):
	    bt = bond.GetBondType()
	    stereo = int(bond.GetStereo())
	    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
	    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
	    return torch.Tensor(fbond + fstereo)








