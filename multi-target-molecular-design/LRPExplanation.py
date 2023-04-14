# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/17 11:20
# @File: LRPExplanation.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import os
from copy import deepcopy

import dgl
import torch
import torch as th
import mtMolDes
from torch import nn
from filter import relevance_filter
from lrp import RelevancePropagationLinear, RelevancePropagationReLU, \
    RelevancePropagationDropout, RelevancePropagationTAGCN
from dgl.nn.pytorch import GraphConv, TAGConv, ChebConv, GATConv


def layers_lookup() -> dict:
    """
    Lookup table to map network layer to associated lrpDes operation.
    Returns:
        Dictionary holding class mappings.
    """
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        dgl.nn.pytorch.conv.tagconv.TAGConv: RelevancePropagationTAGCN
    }
    return lookup_table


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout etc.!

        # Parse solubNet
        self.layers = self._get_layer_operations(model)

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _get_layer_operations(self, model) -> torch.nn.ModuleList:
        """
        Get all network operations and store them in a list.
        Returns:
            Layers of original model stored in module list.
        """
        layers = torch.nn.ModuleList()

        # Parse solubNet
        for layer in model.top_percep_block.gcn_layers:
            layers.append(layer)

        for layer in model.atom_red_block.fc_layers:
            # print('Each layer:', layer)
            layers.append(layer)
        return layers

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.
        Returns:
            LRP-model as module list.
        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()

        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer)
            except KeyError:
                message = f"Layer-wise relevance propagation not implemented for " \
                          f"{layer.__class__.__name__} layer."
                raise NotImplementedError(message)
        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward method that first performs standard inference followed by layer-wise relevance propagation.
        Args:
            x: Input tensor representing a molecule.
        Returns:
            Tensor holding relevance scores.
        """
        activations = list()
        all_relevance = []

        # Run inference and collect activations.
        graph = x
        x = x.ndata['h'].float()
        with torch.no_grad():
            activations.append(torch.ones_like(x))

            for layer in self.layers[:3]:
                x = layer.forward(graph, x)
                activations.append(x)

            for layer in self.layers[3:]:
                x = layer.forward(x)
                activations.append(x)


        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.sum(activations.pop(0))
        # print('The relevance of last layer: ', relevance.shape)
        all_relevance.append(relevance)

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers[:6]):
            relevance = layer.forward(activations.pop(0), relevance)
            # print('The relevance of %d layer: ' %(i), relevance.shape)
            all_relevance.append(relevance)

        for i, layer in enumerate(self.lrp_layers[6:]):
            relevance = layer.forward(activations.pop(0), relevance, graph)
            all_relevance.append(relevance)

        # print(all_relevance[-1])
        return all_relevance


