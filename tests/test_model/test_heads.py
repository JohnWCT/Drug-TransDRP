import pytest
import torch
from transdrp_multilabel.model.heads import MultiOutputDrugHead

def test_multi_output_drug_head():
    head = MultiOutputDrugHead(
        input_dim=16, # concatenated dimension
        output_dim=1,
        hidden_dims=[8, 4],
        drug_num=3,
        drop=0.0
    )

    # Batch size = 4
    x = torch.randn(4, 8) # batch latent representations (latent_dim = 8)
    node_x = torch.randn(3, 8) # drug node features (drug_feat_dim = 8)
    # Fully connected graph between 3 drug nodes
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)

    out = head(x, node_x, edge_index)

    assert out.shape == (4, 3)
    # Check gradient flow
    out.sum().backward()
    assert head.output_layer.weight.grad is not None
