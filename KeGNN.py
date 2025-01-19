import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class KnowledgeEnhancedLayer(nn.Module):
    def __init__(self, input_dim):
        super(KnowledgeEnhancedLayer, self).__init__()
        self.param_P = nn.Parameter(torch.randn(input_dim))
        self.param_K = nn.Parameter(torch.randn(input_dim))
        self.param_U = nn.Parameter(torch.randn(input_dim))

    def forward(self, Z):
        """
        Apply the knowledge-enhancement formula: Y' = \sigma(Z + P \odot K \odot U)
        """
        knowledge_adjustment = self.param_P * self.param_K * self.param_U
        enhanced_output = Z + knowledge_adjustment
        return F.relu(enhanced_output)  # Using ReLU as the activation function

class KeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_type='GCN'):
        super(KeGNN, self).__init__()
        
        if gnn_type == 'GCN':
            self.gnn1 = GCNConv(input_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, output_dim)
        elif gnn_type == 'GATv2':
            self.gnn1 = GATv2Conv(input_dim, hidden_dim)
            self.gnn2 = GATv2Conv(hidden_dim, output_dim)
        else:
            raise ValueError("Unsupported GNN type. Choose 'GCN' or 'GATv2'.")
        
        self.knowledge_layer = KnowledgeEnhancedLayer(output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass for KeGNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges in COO format [2, num_edges]
        
        Returns:
            Output logits after knowledge enhancement
        """
        # Apply GNN layers
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        
        # Apply knowledge enhancement layer
        enhanced_output = self.knowledge_layer(x)
        return F.log_softmax(enhanced_output, dim=1)  # Log softmax for classification

# Example usage
def main():
    num_nodes = 100
    input_dim = 16
    hidden_dim = 32
    output_dim = 10

    # Sample graph data
    x = torch.randn((num_nodes, input_dim))  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random graph edges

    # Initialize and run KeGNN
    model = KeGNN(input_dim, hidden_dim, output_dim, gnn_type='GATv2')
    output = model(x, edge_index)
    print("Output logits:", output)

if __name__ == "__main__":
    main()
