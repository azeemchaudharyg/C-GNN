import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, APPNP, global_mean_pool, BatchNorm
from torch_geometric.nn.aggr import MultiAggregation, AttentionalAggregation

# from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation, StdAggregation, SumAggregation, LSTMAggregation

from utils.data_loader import features_per_node


class CustomSAGE_APPNP(torch.nn.Module):
    def __init__(self, in_channels=features_per_node, hidden_channels=256, out_channels=2, K=10, alpha=0.1, dropout_p=0.5):
        super(CustomSAGE_APPNP, self).__init__()
        
        # Attentional Pooling
        # self.attn_pool = AttentionalAggregation(
        #     gate_nn=torch.nn.Sequential(
        #         torch.nn.Linear(hidden_channels, 1),
        #         torch.nn.Sigmoid()
        #     )
        # )
        
        #aggr_modalities = ['max', 'std']
        
        # self.multi_aggr = MultiAggregation(
        #     aggrs=['max', 'std'],
        #     mode='attn',
        #     mode_kwargs=dict(in_channels=in_channels, out_channels=hidden_channels, num_heads=4)
        # )

        # SAGEConv layer 1 
        self.sage1 = SAGEConv(in_channels, hidden_channels)  # aggr=self.multi_aggr
        self.bn1 = BatchNorm(hidden_channels)

        # APPNP propagation
        self.appnp = APPNP(K=K, alpha=alpha)

        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_p)

        # Fully Connected Classifier Layer
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # SAGEConv Layer 1
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # APPNP Layer
        x = self.appnp(x, edge_index)

        # Graph pooling 
        x = global_mean_pool(x, batch)
        
        # Graph pooling (Attentional Aggregation)
        # x = self.attn_pool(x, batch)

        # Fully Connected Layer
        out = self.lin(x)
        
        return out