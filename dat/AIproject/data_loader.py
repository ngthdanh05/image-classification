import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=32, img_size=128, val_split=0.2):
    # Transform cho ảnh
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load toàn bộ dataset từ PetImages
    full_dataset = datasets.ImageFolder(
        root='data/PetImages', 
        transform=transform
    )
    
    # Tự động chia thành train và validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = full_dataset.classes  # ['Cat', 'Dog']
    
    print(f"Dataset loaded successfully!")
    print(f"Total images   : {len(full_dataset)}")
    print(f"Train images   : {train_size}")
    print(f"Validation images: {val_size}")
    print(f"Classes        : {class_names}")
    
    return train_loader, val_loader, class_names