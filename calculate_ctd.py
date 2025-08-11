from cProfile import label
import os
import numpy as np
import math
import torch
from collections import defaultdict
from typing import Optional, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

class KNN:
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, k: int = 5):
        self.data = data
        self.labels = labels
        self.k = k

    def calculate_rank(self, query: torch.Tensor, query_label: int) -> int:
        # TODO: optimized performance (dimension reduction?)
        # print(query.shape, self.data.shape, self.labels.shape)
        indices = self.get_neighbor(query)

        rank = 0
        for i in range(len(indices)):
            if self.data[indices[i]].equal(query):
                continue
            if self.labels[indices[i]] == query_label:
                return rank
            rank += 1
        
        return len(indices) + 1
    
    def get_neighbor(self, query: torch.Tensor) -> list:
        distances = torch.cdist(query.unsqueeze(0), self.data.unsqueeze(0)).squeeze(0)
        _, indices = torch.topk(distances, self.k, largest=False)
        indices = indices[0]
        return indices.tolist()

# QUESTION: Move this to utils.py?
def get_item_activations(model: torch.nn.Module, input_tensor: torch.Tensor, device: str = 'cpu') -> dict:
    """
    Extract activations from specified layers in a PyTorch model.
    
    Args:
        model: The PyTorch model.
        input_tensor: The input tensor for the forward pass.
        layer_names: List of layer names (strings) to extract activations from.
                     If None, extract from all layers except the root.
        device: Device to run the model on ('cpu' or 'cuda').
        
    Returns:
        activations: dict mapping layer names to their output tensors.
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook

    handles = []
    for name, module in model.named_modules():
        if name == '':
            continue
        handles.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return activations


def get_dataset_activations(model: torch.nn.Module, data_loader: DataLoader, device='cpu') -> Tuple[dict, torch.Tensor]:
    """
    Extract activations from specified layers in a PyTorch model for a dataset.
    
    Args:
        model: The PyTorch model.
        data_loader: DataLoader for the dataset.
        device: Device to run the model on ('cpu' or 'cuda').
        
    Returns:
        all_activations: dict mapping layer names to their output tensors for the entire dataset.
    """
    all_activations = defaultdict(torch.Tensor)
    all_labels = torch.Tensor().to(device)
    model = model.to(device)
    model.eval()

    # print("Getting model activations...")
    for item, label in tqdm(data_loader):
        item = item.to(device)
        label = label.to(device)
        activations = get_item_activations(model, item, device)
        all_labels = torch.cat((all_labels, label), dim=0)
        for layer_name, activation in activations.items():
            if 'conv' in layer_name or 'linear' in layer_name:
                all_activations[layer_name] = torch.cat((all_activations[layer_name], activation), dim=0)

    # Save activations to disk
    # TODO: Maybe save with a name that includes the dataset and model name
    torch.save(all_activations, './cache/activations.pt')
    torch.save(all_labels, './cache/labels.pt')

    return all_activations, all_labels

def get_knn_model(layer_name: str, all_activations: dict, all_labels: torch.Tensor, k: int = 5, retrain: bool = False) -> KNN:
    """
    Constructs a KNN model from the activations of the specified layers in the model.
    
    Args:
        model: The PyTorch model.
        data_loader: DataLoader for the dataset.
        device: Device to run the model on ('cpu' or 'cuda').
        k: Number of nearest neighbors to consider.
        
    Returns:
        knn_model: KNN model constructed from the activations.
    """
    if os.path.exists(f'./cache/knn_model_{layer_name}.pt') and not retrain:
        print(f"Loading saved KNN model for {layer_name}...")
        return torch.load(f'./cache/knn_model_{layer_name}.pt')
    activation = all_activations[layer_name]
    activation = activation.view(activation.size(0), -1)
    knn_model = KNN(activation, all_labels, k=k)
    torch.save(knn_model, f'./cache/knn_model_{layer_name}.pt')
    return knn_model

def construct_knn_graph(knn_model: KNN) -> dict:
    """
    Constructs a KNN graph from the KNN model.
    
    Args:
        knn_model: KNN model constructed from the activations.
        
    Returns:
        knn_graph: Dictionary mapping each node to its neighbors.
    """
    knn_graph = defaultdict(list)
    for i in range(len(knn_model.data)):
        neighbors = knn_model.get_neighbor(knn_model.data[i])
        knn_graph[i].extend(neighbors)
    return knn_graph

def calculate_weight(knn_graph_map: dict, knn_model_map: dict) -> dict:
    knn_model = knn_model_map[list(knn_model_map.keys())[0]]
    index2label = knn_model.labels
    label2index = defaultdict(list)
    for i, label in enumerate(index2label):
        label2index[int(label.item())].append(i)
    
    # m is the total number of edges in the graph
    m = defaultdict(int)
    # E is the number of edges within community i
    # E[layer_name] is a list with length num_classes, where each element corresponds to the number of edges within that class
    E = defaultdict(lambda: torch.zeros(10, dtype=torch.float32))
    # k is the sum of node degrees in community i
    # k[layer_name][label] is the sum of node degrees for label in layer_name
    k = defaultdict(lambda: torch.zeros(10, dtype=torch.float32))

    gamma = 1
    
    for layer_name, knn_graph in knn_graph_map.items():

        knn_model = knn_model_map[layer_name]
        for i in range(len(knn_model.data)):
            try:
                label = knn_model.labels[i]
                # cast label to int
                label = int(label.item())
                m[layer_name] += len(knn_graph[i])
                k[layer_name][label] += len(knn_graph[i])
                for idx in knn_graph[i]:
                    if knn_model.labels[idx] == label:
                        E[layer_name][label] += 1
                    else:
                        E[layer_name][label] += 0
            except Exception as e:
                print(f"Error processing index {i} in layer {layer_name}: {e}")
                print(f"{i} in layer {layer_name} with label {label} and knn_model of size {len(knn_model.data)}")
                continue
    
    # print(f"m: {m}")
    # print(f"E: {E}")
    # print(f"k: {k}")

    weights = defaultdict(lambda: torch.zeros(10, dtype=torch.float32))
    # For each class
    for c in range(10):
        modularities = {}
        for layer_name, knn_graph in knn_graph_map.items():
            edge_count = [E[layer_name][c], sum(E[layer_name]) - E[layer_name][c]]
            degree_sum = [k[layer_name][c], sum(k[layer_name]) - k[layer_name][c]]

            q = 0
            for i in range(2):
                q += m[layer_name] * edge_count[i] - gamma * pow(degree_sum[i] / (2*m[layer_name] + 1e-10), 2)
            modularities[layer_name] = q
        # print(f"Modularities for class {c}: {modularities}")
        Q_max = max(modularities.values())
        Q_min = min(modularities.values())
        for layer_name in knn_graph_map.keys():
            if Q_max - Q_min == 0:
                weights[layer_name][c] = 1.0
            else:
                weights[layer_name][c] = (modularities[layer_name] - Q_min) / (Q_max - Q_min)


    return weights


def preprocessing(model: torch.nn.Module, data_loader: DataLoader, device: str = 'cpu') -> Tuple[dict, dict, dict]:
    """
    Construct KNN models and KNN graphs for each layer in the model.
    Then calculate the weights for each (layer, class) pair based on the KNN graphs.
    Returns:
        knn_graph_map: Dictionary mapping layer names to their KNN graphs.
        weights: Dictionary mapping (layer_name, class_label) to their weights.
    """
    activations, labels = get_dataset_activations(model, data_loader, device)
    knn_graph_map = {}
    knn_model_map = {}
    for layer_name in activations.keys():
        knn_model = get_knn_model(layer_name, activations, labels, retrain=True, k=math.ceil(np.sqrt(len(data_loader.dataset))))
        knn_graph_map[layer_name] = construct_knn_graph(knn_model)
        knn_model_map[layer_name] = knn_model
    weights = calculate_weight(knn_graph_map, knn_model_map)

    # Save the KNN graphs and weights to disk
    torch.save(knn_graph_map, './cache/knn_graph_map.pt')
    torch.save(knn_model_map, './cache/knn_model_map.pt')
    torch.save(dict(weights), './cache/weights.pt')

    return knn_graph_map, knn_model_map, weights


def calculate_cumulative_topological_distance(model: torch.nn.Module, data_loader: DataLoader, weights: dict, device: str, query: torch.Tensor, query_label: torch.Tensor, retrain: bool = False) -> torch.Tensor:
    """
    Calculates the cumulative topological distance (CTD) for each sample in data_loader.
    """

    knn_map = {}
    if not retrain and os.path.exists('./cache/activations.pt') and os.path.exists('./cache/labels.pt') and os.path.exists('./cache/knn_model_map.pt'):
            print("Loading saved activations and labels...")
            all_activations = torch.load('./cache/activations.pt', weights_only=False)
            all_labels = torch.load('./cache/labels.pt', weights_only=False)
            knn_map = torch.load('./cache/knn_model_map.pt', weights_only=False)
    else:
        all_activations, all_labels = get_dataset_activations(model, data_loader, device)
        for layer_name in all_activations.keys():
            knn_map[layer_name] = get_knn_model(layer_name, all_activations, all_labels, retrain=retrain, k=math.ceil(np.sqrt(len(data_loader.dataset))))
    
    model.eval()
    ctd_values = torch.zeros(len(query), len(all_activations.keys()), dtype=torch.float32, device=device)
    arr = list(all_activations.keys())
    layer2idx = {arr[i]: i for i in range(len(arr))}

    if query is not None:
        query = get_item_activations(model, query, device)


    print("Calculating CTD values...")
    for layer_name, activation in all_activations.items():
        if 'conv' in layer_name or 'linear' in layer_name:
            activation = activation.view(activation.size(0), -1)
            knn = knn_map[layer_name]
            for i in tqdm(range(query[layer_name].size(0))):
                q = query[layer_name][i].view(1, -1)
                rank = knn.calculate_rank(q, query_label=query_label[i])
                ctd_values[i][layer2idx[layer_name]] = rank * weights[layer_name][int(query_label[i].item())]
    return ctd_values
