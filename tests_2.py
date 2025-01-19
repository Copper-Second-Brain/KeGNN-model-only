import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.datasets import Planetoid
from matplotlib import pyplot as plt
from EnhancedGATv2 import EnhancedGATv2
from KeGNN import KeGNN

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATv2Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATv2Model, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCNWithKE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ke_dim):
        super(GCNWithKE, self).__init__()
        self.conv1 = GCNConv(input_dim + ke_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, ke):
        x = torch.cat([x, ke], dim=1)
        x = self.dropout(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load Planetoid dataset
data = Planetoid(root='./data', name='Cora')[0]
data = data.to(torch.device('cpu'))

# Generate knowledge embeddings (for demonstration purposes)
ke_dim = 2
knowledge_embeddings = torch.rand((data.num_nodes, ke_dim))

# Define dimensions
input_dim = data.num_features
hidden_dim = 16
output_dim = len(data.y.unique())

# Initialize models
gcn_model = GCNModel(input_dim, hidden_dim, output_dim)
gatv2_model = GATv2Model(input_dim, hidden_dim, output_dim)
gcn_with_ke_model = GCNWithKE(input_dim, hidden_dim, output_dim, ke_dim)
enhanced_gatv2_model = EnhancedGATv2(input_dim, hidden_dim, output_dim, logical_rules=[1, 2])
kegnn_gcn_model = KeGNN(input_dim, hidden_dim, output_dim, gnn_type='GCN')
kegnn_gat_model = KeGNN(input_dim, hidden_dim, output_dim, gnn_type='GATv2')

# Train and evaluate

def train_and_evaluate(model, data, ke=None, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if ke is not None:
            out = model(data.x, data.edge_index, ke)
        else:
            out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            if ke is not None:
                out = model(data.x, data.edge_index, ke)
            else:
                out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask] == data.y[data.test_mask]
            accuracy = int(correct.sum()) / int(data.test_mask.sum())
            best_accuracy = max(best_accuracy, accuracy)
        model.train()

    return best_accuracy

gcn_acc = train_and_evaluate(gcn_model, data)
gatv2_acc = train_and_evaluate(gatv2_model, data)
gcn_with_ke_acc = train_and_evaluate(gcn_with_ke_model, data, ke=knowledge_embeddings)
enhanced_gatv2_acc = train_and_evaluate(enhanced_gatv2_model, data)
kegnn_gcn_acc = train_and_evaluate(kegnn_gcn_model, data)
kegnn_gat_acc = train_and_evaluate(kegnn_gat_model, data)

# Compare results
print("GCN Accuracy:", gcn_acc)
print("GATv2 Accuracy:", gatv2_acc)
print("GCN with KE Accuracy:", gcn_with_ke_acc)
print("Enhanced GATv2 Accuracy:", enhanced_gatv2_acc)
print("KeGNN (GCN) Accuracy:", kegnn_gcn_acc)
print("KeGNN (GATv2) Accuracy:", kegnn_gat_acc)

# Plot the results
plt.bar([
    'GCN', 'GATv2', 'GCN with KE', 'Enhanced GATv2', 'KeGNN (GCN)', 'KeGNN (GATv2)'
], [
    gcn_acc, gatv2_acc, gcn_with_ke_acc, enhanced_gatv2_acc, kegnn_gcn_acc, kegnn_gat_acc
])
plt.ylabel('Accuracy')
plt.title('Model Comparison: GCN, GATv2, GCN with KE, Enhanced GATv2, KeGNN (GCN), KeGNN (GATv2)')
plt.show()
