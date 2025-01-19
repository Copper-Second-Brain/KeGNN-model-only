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

# Load Planetoid dataset
data = Planetoid(root='./data', name='Cora')[0]
data = data.to(torch.device('cpu'))

# Define dimensions
input_dim = data.num_features
hidden_dim = 16
output_dim = len(data.y.unique())

# Initialize models
kegnn_model = KeGNN(input_dim, hidden_dim, output_dim, gnn_type='GATv2')
gcn_model = GCNModel(input_dim, hidden_dim, output_dim)
gatv2_model = GATv2Model(input_dim, hidden_dim, output_dim)
enhanced_gatv2_model = EnhancedGATv2(input_dim, hidden_dim, output_dim, logical_rules=[1, 2])

# Train and evaluate

def train_and_evaluate(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask] == data.y[data.test_mask]
            accuracy = int(correct.sum()) / int(data.test_mask.sum())
            best_accuracy = max(best_accuracy, accuracy)
        model.train()

    return best_accuracy

kegnn_acc = train_and_evaluate(kegnn_model, data)
gcn_acc = train_and_evaluate(gcn_model, data)
gatv2_acc = train_and_evaluate(gatv2_model, data)
enhanced_gatv2_acc = train_and_evaluate(enhanced_gatv2_model, data)

# Compare results
print("KeGNN Accuracy:", kegnn_acc)
print("GCN Accuracy:", gcn_acc)
print("GATv2 Accuracy:", gatv2_acc)
print("Enhanced GATv2 Accuracy:", enhanced_gatv2_acc)

# Plot the results
plt.bar(['KeGNN', 'GCN', 'GATv2', 'Enhanced GATv2'], [kegnn_acc, gcn_acc, gatv2_acc, enhanced_gatv2_acc])
plt.ylabel('Accuracy')
plt.title('KeGNN vs GCN vs GATv2 vs Enhanced GATv2')
plt.show()
