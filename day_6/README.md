# Graph Neural Networks (GNNs) - Day 6 PyTorch Learning Journey

## 🧠 Project Overview

This project implements and explores **Graph Neural Networks (GNNs)** using PyTorch Geometric, focusing on node classification tasks. We specifically work with **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** on citation network datasets.

## 🎯 What We're Doing

We are implementing and training different types of Graph Neural Networks to perform **node classification** on citation networks. The main goal is to predict the research field/category of academic papers based on:
- Paper content (features)
- Citation relationships (graph structure)
- Neighboring papers' information

## 📊 Dataset: CiteSeer

We use the **CiteSeer** dataset from the Planetoid collection:
- **Nodes**: Academic papers (3,327 papers)
- **Edges**: Citation relationships between papers
- **Features**: Bag-of-words representation of paper content
- **Labels**: Research field categories (6 classes)
- **Task**: Semi-supervised node classification

## 🔬 The Ideology Behind Graph Neural Networks

### Why GNNs Matter

Traditional neural networks work well with grid-like data (images, sequences), but real-world data often comes in graph structures:
- Social networks (users connected by friendships)
- Citation networks (papers connected by citations)
- Molecular structures (atoms connected by bonds)
- Knowledge graphs (entities connected by relationships)

### Core Concepts

#### 1. **Message Passing Framework**
GNNs follow a message-passing paradigm:
```
For each node:
1. Collect messages from neighbors
2. Aggregate these messages
3. Update node representation
4. Repeat for multiple layers
```

#### 2. **Inductive Bias**
GNNs encode the assumption that:
- **Homophily**: Connected nodes tend to be similar
- **Local Structure**: A node's representation should depend on its neighborhood
- **Permutation Invariance**: Node ordering shouldn't matter

#### 3. **Mathematical Foundation**
For a node `v` at layer `l+1`:
```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```
Where:
- `h_v^(l)` is the representation of node `v` at layer `l`
- `N(v)` is the neighborhood of node `v`
- `UPDATE` and `AGGREGATE` are learnable functions

## 🏗️ Architectures Implemented

### 1. Graph Convolutional Network (GCN)

**Key Idea**: Learn node representations by aggregating features from local neighborhoods.

**How it works**:
- Each layer performs a localized convolution operation
- Messages are aggregated using mean pooling
- Applies a linear transformation followed by activation

**Mathematical Formula**:
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```
Where:
- `A` is the adjacency matrix with self-loops
- `D` is the degree matrix
- `H^(l)` are node features at layer `l`
- `W^(l)` are learnable weights
- `σ` is an activation function

**Advantages**:
- Simple and effective
- Computationally efficient
- Good baseline for graph tasks

**Limitations**:
- All neighbors treated equally
- Can suffer from over-smoothing
- Limited expressiveness

### 2. Graph Attention Network (GAT)

**Key Idea**: Learn to pay different amounts of attention to different neighbors.

**How it works**:
- Computes attention weights for each neighbor
- Weighs neighbor contributions based on learned importance
- Allows multiple attention heads for richer representations

**Mathematical Formula**:
```
α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
h_i' = σ(Σ_j α_ij W h_j)
```
Where:
- `α_ij` is the attention weight from node `j` to node `i`
- `a` is a learnable attention mechanism
- `||` denotes concatenation

**Advantages**:
- Dynamic attention mechanism
- Better handling of heterogeneous neighborhoods
- More interpretable (attention weights)
- Multiple attention heads for different aspects

**Limitations**:
- Higher computational complexity
- More parameters to tune
- Potential overfitting on small datasets

## 🔧 Implementation Details

### Libraries Used
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph deep learning extension
- **NetworkX**: Graph visualization and analysis
- **Matplotlib**: Plotting and visualization

### Model Architecture

#### GCN Implementation
```python
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

#### GAT Implementation
```python
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, dropout=0.6)
        self.dropout = torch.nn.Dropout(0.6)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### Training Process

1. **Data Loading**: Load CiteSeer dataset with train/validation/test splits
2. **Model Initialization**: Create GCN or GAT model with appropriate dimensions
3. **Optimization**: Use Adam optimizer with learning rate scheduling
4. **Training Loop**: 
   - Forward pass through the model
   - Compute cross-entropy loss on labeled nodes
   - Backpropagation and parameter updates
   - Evaluation on validation set
5. **Testing**: Final evaluation on test set

### Evaluation Metrics
- **Accuracy**: Percentage of correctly classified nodes
- **Loss**: Cross-entropy loss for multi-class classification
- **Training/Validation Curves**: Monitor overfitting

## 🚀 How Everything Works Together

### 1. Graph Representation
- Papers → Nodes with feature vectors
- Citations → Edges connecting related papers
- Labels → Ground truth categories for supervised learning

### 2. Feature Propagation
- Node features propagate through the graph structure
- Each layer aggregates information from broader neighborhoods
- Deep networks capture multi-hop relationships

### 3. Semi-Supervised Learning
- Only a small fraction of nodes are labeled (train set)
- Models learn to classify unlabeled nodes using:
  - Their own features
  - Graph structure
  - Labeled neighbors' information

### 4. Inductive Learning
- Models learn general principles for node classification
- Can potentially generalize to new, unseen graphs
- Captures both local and global graph patterns

## 📈 Expected Results

### Performance Comparison
- **GCN**: ~70-75% accuracy on CiteSeer
- **GAT**: ~72-78% accuracy on CiteSeer (with attention mechanism)

### Key Observations
1. **Graph structure matters**: Significant improvement over node-feature-only baselines
2. **Attention helps**: GAT often outperforms GCN by learning adaptive neighbor weights
3. **Depth vs. Performance**: Deeper networks may suffer from over-smoothing
4. **Semi-supervised power**: Good performance with limited labeled data

## 🔍 Key Insights and Learnings

### Why GNNs Work
1. **Homophily**: Similar papers tend to cite each other
2. **Structural Information**: Citation patterns reveal research communities
3. **Feature Smoothing**: Averaging neighbor features reduces noise
4. **Multi-scale Learning**: Different layers capture different relationship scales

### Challenges Addressed
1. **Irregular Graph Structure**: Unlike images/sequences, graphs have no fixed structure
2. **Varying Neighborhood Sizes**: Different nodes have different numbers of neighbors
3. **Scalability**: Efficient computation on large graphs
4. **Limited Labels**: Learning with sparse supervision

### Future Extensions
- **Graph Transformer**: Self-attention mechanisms for graphs
- **GraphSAGE**: Sampling-based approaches for large graphs
- **Graph LSTM**: Temporal graph neural networks
- **Heterogeneous Graphs**: Multiple node/edge types

## 🛠️ Running the Code

1. **Install Dependencies**:
   ```bash
   pip install torch torch-geometric networkx matplotlib
   ```

2. **Run Training**:
   - Open `Training_of_GNN.ipynb`
   - Execute cells sequentially
   - Compare GCN vs GAT performance

3. **Experiment with**:
   - Different hidden dimensions
   - Various attention heads (for GAT)
   - Different datasets (Cora, PubMed)
   - Layer depths and dropout rates

## 📚 Further Reading

- **Papers**:
  - "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
  - "Graph Attention Networks" (Veličković et al., 2018)
  - "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)

- **Resources**:
  - PyTorch Geometric Documentation
  - CS224W: Machine Learning with Graphs (Stanford)
  - Graph Neural Networks: A Review of Methods and Applications

## 🎯 Project Goals Achieved

✅ **Understanding GNN Fundamentals**: Message passing, aggregation, update functions  
✅ **Implementing Core Architectures**: GCN and GAT from scratch concepts  
✅ **Hands-on Experience**: Training on real citation network data  
✅ **Performance Analysis**: Comparing different GNN approaches  
✅ **Graph Deep Learning Pipeline**: End-to-end implementation  

This project provides a solid foundation for understanding how Graph Neural Networks work and their applications in real-world scenarios involving graph-structured data. 