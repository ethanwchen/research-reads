## GraphSAGE - Inductive Representation Learning on Large Graphs (2017)

This focuses on inductive node embedding. Unlike traditional embedding appraoches that are based on matrix factorization, it leverages node features in order to learn an embedding function that generalizes to unseen nodes.

### The Problem

Most traditional graph neural networks are transductive, meaning they can only make predictions on nodes seen during training. However, in real-world systems, new nodes appear all the time. You don't want to retrain your entire model from scratch every time that happens. 

GraphSAGE solves the issue with an inductive method, learning a function that generates node embeddings on the fly, even for previously unseen nodes.

### The Idea

Instead of learning unique embeddings for every node, GraphSAGE learns how to aggregate features from a node's neighbors. You get some learned funtion(node features, neighbor features) -> embedding.

### Model Architecture

1. Initial Features
Each node starts with an input feature vector:
- Can be raw features (like merchant category, transaction volume, etc...)
- Or one-hot encodings if no features available
- We can start by calling $h_{v}^{0}$ the initial embedding of node v

2. Neighborhood Sampling
Instead of using all neighbors, GraphSAGE samples a fixed size neighborhood N(v) for each node. This keeps the computation manageable and enables mini-batching.

3. Aggregation Step
At each layer k, you:
- Aggregate the representations from neighbors N(v)
- Combine them with the node's own representation

4. Layer-Wise Propagation
You apply multiple layers of aggregation:
- Layer 1: aggregates info from 1-hop neighbors
- Layer 2: aggregates info from 2-hop neighbors
- ...

### Aggregation Functions
GraphSAGE explores different types of aggregation:
1. Mean Aggregation
2. LSTM Aggregator
- Sequence based aggregator using an LSTM
- Sensitive to the order of neighbors
3. Pooling Aggregator
- Applies a fully-connected network followed by a max-pool operation

### Training
GraphSAGE supports different objectives
- Supervised node classifications: train on known fraud labels, merchant type, etc...
- Unsupervised embedding learning: use random walks + negative sampling to maximize similarity between nearby nodes
