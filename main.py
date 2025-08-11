import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import math
from pyod.models.pca import PCA

from dataset import IMG_Dataset
from resnet import resnet20
from calculate_ctd import calculate_cumulative_topological_distance, preprocessing

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    poison_data_dir = 'data/cifar10/test_poisoned/img'
    poison_label_path = 'data/cifar10/test_poisoned/labels'
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    poison_dataset = IMG_Dataset(poison_data_dir, poison_label_path, transform)
    poison_dataset_labels = torch.load(poison_label_path, weights_only=True)
    subset = Subset(poison_dataset, range(1000))
    poison_dataloader = DataLoader(
        subset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
    )
    poison_labels = [poison_dataset_labels[i] for i in subset.indices]

    clean_data_dir = 'data/cifar10/test/img'
    clean_label_path = 'data/cifar10/test/labels'
    clean_dataset = IMG_Dataset(clean_data_dir, clean_label_path, transform)

    clean_dataset_labels = torch.load(clean_label_path, weights_only=True)
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(clean_dataset_labels):
        if len(class_indices[int(label)]) < 200:
            class_indices[int(label)].append(idx)

    subset_indices = [i for indices in class_indices.values() for i in indices]
    clean_subset = Subset(clean_dataset, subset_indices)

    clean_subset_labels = [clean_dataset_labels[i] for i in clean_subset.indices]
    clean_subset_samples = [clean_dataset[i][0] for i in clean_subset.indices]
    clean_subset_samples = torch.stack(clean_subset_samples).to(device)

    clean_dataloader = DataLoader(
        clean_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True
    )

    clean_subset_dataloader = DataLoader(
        clean_subset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True
    )

    model = resnet20(num_classes=10)
    model.load_state_dict(torch.load('models/resnet20_aug.pt', map_location=device, weights_only=True))
    model = model.to(device)

    print("Model initialized.")

    knn_graph_map, knn_model_map, weights = preprocessing(model, clean_subset_dataloader, device)
    print("Preprocessing completed.")

    ctd_clean = calculate_cumulative_topological_distance(model, clean_subset_dataloader, weights, device, query=clean_subset_samples, query_label=clean_subset_labels, retrain=True).clone().detach()
    # Cluster the ctd_clean values by labels and save in cache
    ctd_clean_clusters = {i: [] for i in range(10)}
    for idx, label in enumerate(clean_subset_labels):
        ctd_clean_clusters[int(label)].append(ctd_clean[idx])
    torch.save(ctd_clean_clusters, 'cache/ctd_clean_clusters.pt')

    torch.save(ctd_clean, 'cache/ctd_clean.pt')

    poison_labels = [poison_dataset_labels[i] for i in subset.indices]
    poison_samples = [poison_dataset[i][0] for i in subset.indices]
    poison_samples = torch.stack(poison_samples).to(device)
    ctd_poisoned = calculate_cumulative_topological_distance(model, clean_subset_dataloader, weights, device, query=poison_samples, query_label=poison_labels, retrain=True).clone().detach()
    ctd_poisoned_clusters = {i: [] for i in range(10)}
    for idx, label in enumerate(poison_labels):
        ctd_poisoned_clusters[int(label)].append(ctd_poisoned[idx])
    torch.save(ctd_poisoned_clusters, 'cache/ctd_poisoned_clusters.pt')

    # To get the labels of a Subset object, access the underlying dataset and use the indices
    # print("Cumulative Topological Distance (poisoned):", ctd_poisoned.mean().item(), "±", ctd_poisoned.std().item())
    torch.save(ctd_poisoned, 'cache/ctd_poisoned.pt')

    # clf = PCA(contamination=0.01)
    # clf.fit(ctd_clean.reshape(-1, 1).cpu().numpy())
    # ctd_clean_scores = clf.decision_scores_
    # print("PCA decision scores for clean data:", ctd_clean_scores.mean(), "±", ctd_clean_scores.std())

    # pred = clf.predict(ctd_poisoned.reshape(-1, 1).cpu().numpy())
    # print(sum(pred)/len(pred), "of the poisoned data is predicted as outliers by PCA.")
