Module jaqpotpy_torch.featurizers.featurizer
============================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`Featurizer()`
:   Abstract base class for featurizers.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * jaqpotpy_torch.featurizers.smiles_graph_featurizer.SmilesGraphFeaturizer

    ### Methods

    `featurize(self, *args, **kwargs)`
    :   Abstract method to featurize the input data.
        
        Returns:
            The featurized data.