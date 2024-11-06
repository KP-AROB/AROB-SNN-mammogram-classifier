import os, torch
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from torchvision import transforms
from src.utils.datasets import CustomImageFolderDataset
from torch.utils.data import random_split

# TODO: Set encoder as variable
def load_image_folder_dataloader(
        root,
        image_size, 
        image_encoder,
        batch_size: int = 16, 
        intensity : int = 128, 
        gpu : bool = True
    ):
    """
    Retrieves an Image Folder Dataset and 
    returns torch dataloader for training and testing

    Parameters :
    ----------------

    :root: The root folder to find class directories
    :image_size: The input image size
    :batch_size: The batch size
    :time: The simulation during which the input was encoded
    :dt: The simulation time step
    :intensity: Pixel intensity multiplier
    :gpu: Whether or not the gpu is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:

    """
    n_workers = gpu * 4 * torch.cuda.device_count()
    
    dataset = CustomImageFolderDataset(
        image_encoder,
        None,
        root=root,
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(), 
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda x: x * intensity)
            ]
        ),
    )

    print("\n# Available labels in dataset :", dataset.class_to_idx)

    img_nums = int(len(dataset))
    valid_num = int(img_nums * 0.2)
    train_num = img_nums - valid_num
    train_data, val_data = random_split(dataset, [train_num, valid_num])

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=gpu,
    )
    
    return train_dataloader, val_dataloader

def load_mnist_dataloader(
        data_dir: str,
        image_size, 
        encoder,
        batch_size: int = 16, 
        intensity : int = 128, 
        gpu : bool = True):

    """
    Retrieves the MNIST Dataset and 
    returns torch dataloader for training and testing

    Parameters :
    ----------------

    :image_size: The input image size
    :batch_size: The batch size
    :time: The simulation during which the input was encoded
    :dt: The simulation time step
    :intensity: Pixel intensity multiplier
    :gpu: Whether or not the gpu is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:

    """
    n_workers = gpu * 4 * torch.cuda.device_count()

    train_dataset = MNIST(
        encoder,
        None,
        root=os.path.join(data_dir, "MNIST"),
        download=True,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Resize(image_size), 
                transforms.Lambda(lambda x: x * intensity)
            ]
        ),
    )
    test_dataset = MNIST(
        encoder,
        None,
        root=os.path.join(data_dir, "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Resize(image_size), 
                transforms.Lambda(lambda x: x * intensity)
            ]
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    return train_dataloader, test_dataloader