import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import random_split

import os

def get_mnist(data_path: str = "./data"):
    """This function downloads the MNIST dataset into the `data_path`
    directory if it is not there already. WE construct the train/test
    split by converting the images into tensors and normalising them"""

    # transformation to convert images to tensors and apply normalisation
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # prepare train and test set
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset
    
def prepare_dataset(num_partitions: int, val_ratio: float = 0.1):
    """This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each traininset partition
    into train and validation. The test set is left intact and will
    be used by the central server to asses the performance of the
    global model."""

    # get the MNIST dataset
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create train+val for each clients
    train_clients = []
    val_clients = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        train_clients.append(for_train)
        val_clients.append(for_val)

    return train_clients, val_clients, testset

num_resource = 2
train_clients, val_clients, testset = prepare_dataset(
        num_resource, val_ratio = 0.1
)
data_path  = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

for i in range(len(train_clients)):
    torch.save(train_clients[i], data_path+'/train_data_client'+str(i)+'.pt')
    torch.save(val_clients[i], data_path+'/test_data_client'+str(i)+'.pt')
print("generate private data for each clients... Success!")
    
torch.save(testset, data_path+'/test_data_server.pt')
print("generate test data for server... Success!")
