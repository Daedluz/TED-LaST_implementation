# Generate 1000 poisoned images from a clean dataset using poison_transform

import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import supervisor
import config

# Settings (customize as needed)
clean_img_dir = './clean_set/cifar10/test_split/data'  # Folder with clean images (e.g., PNG files)
output_dir = './poisoned_images'          # Where to save poisoned images
label_path = './clean_set/cifar10/test_split/labels'           # Path to torch label file (if available)
num_images = 1000                       # Number of images to process

# Load labels (if available)
labels = torch.load(label_path) if os.path.exists(label_path) else None

data_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])

# Custom dataset for loading images and labels
class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, labels, num_images, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{idx}.png')
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] if self.labels is not None else 0
        return img, label

# Construct dataloader
dataset = SimpleImageDataset(clean_img_dir, labels, num_images, transform=data_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Setup poison transform
poison_transform = supervisor.get_poison_transform(
    poison_type='adaptive_blend',
    dataset_name='cifar10',
    target_class=config.target_class['cifar10'],
    trigger_transform=data_transform,
    is_normalized_input=True,
    alpha=0.3,
    trigger_name='hellokitty_32.png',
    args=None
)

# Make output directory
os.makedirs(output_dir, exist_ok=True)

poisoned_labels = []
img_idx = 0

for batch_imgs, batch_labels in dataloader:
    # batch_imgs: [batch_size, C, H, W], batch_labels: [batch_size]
    poisoned_imgs, poisoned_batch_labels = poison_transform.transform(batch_imgs, batch_labels)
    for i in range(poisoned_imgs.size(0)):
        save_image(poisoned_imgs[i].cpu(), os.path.join(output_dir, f'{img_idx}.png'))
        poisoned_labels.append(poisoned_batch_labels[i].item())
        img_idx += 1

# Save poisoned labels
torch.save(torch.tensor(poisoned_labels), os.path.join(output_dir, 'labels'))
print(f'Saved {img_idx} poisoned images and labels to {output_dir}')
