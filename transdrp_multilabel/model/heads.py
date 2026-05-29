import torch
import torch.nn as nn
from copy import deepcopy
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.data import Data, Batch

class MultiOutputDrugHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] = None,
        drug_num: int = 9,
        drop: float = 0.1,
        act_fn = nn.SELU,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.drop = drop
        self.node_num = drug_num

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append((
                GATConv(hidden_dims[i], hidden_dims[i+1], add_self_loops=True, heads=2, concat=False),
                'x, edge_index -> x'
            ))
            modules.append(act_fn())
            modules.append(nn.Dropout(self.drop))

        self.convseq = Sequential('x, edge_index', modules)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.hidden_dims = hidden_dims
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, node_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch_Size, latent_dim] (e.g. [64, 64])
        # node_x shape: [N_drugs, drug_feat_dim] (e.g. [N, 64])
        # edge_index shape: [2, E]

        # 1. Map features to drug nodes: repeat and transpose
        batch_size = x.size(0)
        x_repeated = x.unsqueeze(1).repeat(1, self.node_num, 1)  # [Batch_Size, N_drugs, latent_dim]
        node_x_repeated = node_x.unsqueeze(0).repeat(batch_size, 1, 1)  # [Batch_Size, N_drugs, drug_feat_dim]

        feat = torch.cat((x_repeated, node_x_repeated), dim=-1)  # [Batch_Size, N_drugs, latent_dim + drug_feat_dim]

        # 2. Build batched graph using PyG Batch directly (much faster and cleaner)
        data_list = [Data(x=feat[i], edge_index=edge_index) for i in range(batch_size)]
        batch_graph = Batch.from_data_list(data_list).to(x.device)

        # 3. Propagate GNN
        embed = self.convseq(batch_graph.x, batch_graph.edge_index)
        embed = embed.reshape([batch_size, self.node_num, self.hidden_dims[-1]])

        # 4. Final projection
        output = self.output_layer(embed)  # [Batch_Size, N_drugs, 1]
        return output.squeeze(dim=-1)      # [Batch_Size, N_drugs]
