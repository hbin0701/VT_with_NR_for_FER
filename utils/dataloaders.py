from torchvision import transforms
from torch.utils.data import DataLoader
from .fer_datasets import FerplusDataset, AffectnetDataset, RafdbDataset

def create_data_loaders(batch_size:int, resize:bool, data_dir:str, dataset:str, aligned=None):

    if dataset == "ferplus":
        dataset = FerplusDataset

    elif dataset == "affectnet":
        dataset = AffectnetDataset

    elif dataset == "rafdb":
        dataset = RafdbDataset

    else:
        print("No matching database was found.")
        raise(AssertionError)

    custom_transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(45),
              transforms.RandomAffine(degrees=15, scale=[1.2, 1.2]),
              transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            ),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    
    test_transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(45),
              transforms.RandomAffine(degrees=15, scale=[1.2, 1.2]),
              transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            ),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

    # For RAF-DB
    if aligned:
        train_data = dataset(data_dir=data_dir, mode="train", aligned=aligned, transforms=custom_transform)
        # valid_data = dataset(data_dir = directory, mode="valid", transforms=custom_transform)
        test_data = dataset(data_dir=data_dir, mode="test", aligned=aligned, transforms=test_transform)

    else:
        train_data = dataset(data_dir = data_dir, mode="train", transforms=custom_transform)
        # valid_data = dataset(data_dir = directory, mode="valid", transforms=custom_transform)
        test_data = dataset(data_dir = data_dir,  mode="test", transforms=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # return train_loader, valid_loader, test_loader
    return train_loader ,test_loader
