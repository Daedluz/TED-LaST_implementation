import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import math
from pyod.models.pca import PCA

from dataset import IMG_Dataset
from resnet import resnet20
from calculate_ctd import calculate_cumulative_topological_distance, preprocessing

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--poison_data_dir", type=str, default="data/cifar10/test_poisoned/img")
    parser.add_argument("--poison_label_path", type=str, default="data/cifar10/test_poisoned/labels")
    parser.add_argument("--clean_data_dir", type=str, default="data/cifar10/test/img")
    parser.add_argument("--clean_label_path", type=str, default="data/cifar10/test/labels")
    parser.add_argument("--weights_path", type=str, default="models/resnet20_aug.pt", help="Path to the model weights")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DataLoader")
    parser.add_argument("--retrain", action="store_true", help="Whether to retrain the model")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    poison_data_dir = args.poison_data_dir
    poison_label_path = args.poison_label_path
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

    clean_data_dir = args.clean_data_dir
    clean_label_path = args.clean_label_path
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
        clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    clean_subset_dataloader = DataLoader(
        clean_subset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    model = resnet20(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    model = model.to(device)

    print("Model initialized.")

    knn_graph_map, knn_model_map, weights = preprocessing(model=model, data_loader=clean_subset_dataloader, device=device, retrain=args.retrain)
    print("Preprocessing completed.")

    ctd_clean = calculate_cumulative_topological_distance(model, clean_subset_dataloader, weights, device, query=clean_subset_samples, query_label=clean_subset_labels, retrain=args.retrain).clone().detach()
    # Cluster the ctd_clean values by labels and save in cache
    ctd_clean_clusters = {i: [] for i in range(args.num_classes)}
    for idx, label in enumerate(clean_subset_labels):
        ctd_clean_clusters[int(label)].append(ctd_clean[idx])
    torch.save(ctd_clean_clusters, 'cache/ctd_clean_clusters.pt')

    # torch.save(ctd_clean, 'cache/ctd_clean.pt')

    poison_labels = [poison_dataset_labels[i] for i in subset.indices]
    poison_samples = [poison_dataset[i][0] for i in subset.indices]
    poison_samples = torch.stack(poison_samples).to(device)
    ctd_poisoned = calculate_cumulative_topological_distance(model, clean_subset_dataloader, weights, device, query=poison_samples, query_label=poison_labels, retrain=True).clone().detach()
    ctd_poisoned_clusters = {i: [] for i in range(args.num_classes)}
    for idx, label in enumerate(poison_labels):
        ctd_poisoned_clusters[int(label)].append(ctd_poisoned[idx])
    torch.save(ctd_poisoned_clusters, 'cache/ctd_poisoned_clusters.pt')

    # torch.save(ctd_poisoned, 'cache/ctd_poisoned.pt')
