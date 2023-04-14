import torch as th
import torch.nn as nn
from .TopPercepBlock import TopPercepBlock
from .AtomRedBlock import AtomRedBlock

class PropPredBlock(nn.Module):
    """The block unit to predict a property for a molecule."""

    def __init__(self, num_features, features_str):
        """Initialize the class.

        Args:
            num_features (int): The dimension of features for all atoms.            
            feature_str (str):  The string to access the atom features.
        """
        super(PropPredBlock, self).__init__()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Begin of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        num_top_block_labels = 50
        num_props = 1
        self.top_percep_block = TopPercepBlock(num_features, num_top_block_labels, features_str)
        self.atom_red_block = AtomRedBlock(num_top_block_labels, num_props)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  End of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def forward(self, graph):
        """Forward function.

        The input and output is a :math:`N_\text{atom}\times N_\text{feature}` matrix and a float number, respectively.

        Args:
            graph (dgl graph): The graphs input to the layer.        

        Returns:
           The predicted property.
        """
        # graph: num_atoms x num_features
        h = self.top_percep_block(graph)  # num_atoms x num_top_block_labels
        h = self.atom_red_block(h)  # 1 x num_props
        return h
    
    def save(self, fn_prefix):
        """Save the model for later use.
        
        Args:
            fn_prefix (str): The model will be saved as fn_prefix.pkl .
        """
        self.top_percep_block.save(fn_prefix+"-top.pkl")
        self.atom_red_block.save(fn_prefix+"-atm.pkl")

    def load(self, fn_prefix):
        """Load the model from a file.
        
        Args:
            fn_prefix (str): The file prefix to load into the model.
        """
        self.top_percep_block.load(fn_prefix+"-top.pkl")
        self.atom_red_block.load(fn_prefix+"-atm.pkl")        

    def to(self, device):
        """Store the model to device (CPU or GPU)

        Args:
            device (str): The device (CPU or GPU) to store the model.
        """
        self.top_percep_block.to(device)
        self.atom_red_block.to(device)
