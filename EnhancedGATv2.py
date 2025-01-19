import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class EnhancedGATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, logical_rules):
        super(EnhancedGATv2, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, out_channels, heads=1, dropout=0.6)
        self.knowledge_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_channels * 8, hidden_channels * 8) for _ in logical_rules
        ])

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        for layer in self.knowledge_layers:
            x = F.elu(layer(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
