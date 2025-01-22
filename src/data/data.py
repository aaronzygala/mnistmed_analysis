import medmnist
import PIL
import torch.utils.data as data
import torchvision.transforms as transforms

def initialize_dataset(data_flag: str, download: bool, resize: bool, as_rgb:bool, size: int, batch_size: int) -> list[data.DataLoader]:
    info = medmnist.INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    print('==> Preparing data...')

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
     
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size)

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return [
        train_loader,
        train_loader_at_eval,
        val_loader,
        test_loader
    ]