import socket
from threading import Thread
import time
import json
import hashlib
import struct
from json import JSONEncoder
import hashlib

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F

ADDRESS = ('0.0.0.0', 9090)

g_socket_server = None # listening socket

g_conn_pool = {}  # connect pool
metaInfo_clients = {}
trained_model = {}

num_actived_clients = 0
num_resource = 2
epochs = 5

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NpEncoder, self).default(obj)

def init():
    """
    initial server side
    """
    global g_socket_server
    g_socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    g_socket_server.bind(ADDRESS)
    g_socket_server.listen(5)  # max waiting queue length
    print("server start! wait for client connecting...")

def accept_client():
    """
    accept new connection
    """
    global num_actived_clients
    while num_actived_clients < num_resource:
        client, info = g_socket_server.accept()  # block until clients connect
        print("connect success")

        # assign unique id to client
        jd_init = {}
        jd_init['id'] = str(info) # id is the ip address
        jd_init['connect_status'] = "connect server successfully!"
        jsonstr = json.dumps(jd_init)
        client.sendall(jsonstr.encode('utf8')) 
        # get meta infor from client
        bytes = client.recv(1024)
        msg = bytes.decode(encoding='utf8')
        jd = json.loads(msg)
        data_size = jd['data_size']
        client_type = jd['client_type']

        g_conn_pool[client_type] = client
        metaInfo_clients[client_type] = {'data_size': data_size}
        print('on client connect: ' + client_type, info)
        num_actived_clients += 1

def remove_client(client_type):
    client = g_conn_pool[client_type]
    if None != client:
        client.close()
        g_conn_pool.pop(client_type)
        print("client offline: " + client_type)
        num_actived_clients -= 1

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

        #train_clients.append(list(for_train))
        #val_clients.append(list(for_val))
        train_clients.append([(i[0].tolist(), i[1]) for i in for_train])
        val_clients.append([(i[0].tolist(), i[1]) for i in for_train])

    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=128)

    return train_clients, val_clients, testloader

def send_data(client, data_type, data):
    """
    Send message for large scale data avoiding Stick Package
    """
    try:
        total_res_bytes = len(data)
        print(total_res_bytes)
        head_dic = {'time': time.localtime(time.time()),
                    'size': total_res_bytes, 
                    'MD5': hashlib.new('md5', data).hexdigest(),
                    'file_name': data_type}
        head_dic_bytes = json.dumps(head_dic).encode('utf-8')
        # print(len(head_dic_bytes))
        print(head_dic)
        head = struct.pack('i', len(head_dic_bytes))
        client.send(head)
        client.send(head_dic_bytes)
        print("data")
        client.send(data)
    except Exception as e:
        print("send data error", e)
        
def get_data(client):
    """
    Handle message for large scale data
    """
    try:
        head = client.recv(4)
        if len(head) == 0:
            print("connection closed!")
            return None
        dic_length = struct.unpack('i', head)[0]
        # print(dic_length)
        head_dic = client.recv(int(dic_length))
        dic = json.loads(head_dic.decode('utf-8'))
        content_length = dic['size']
        print(dic['MD5'])
        content = b''
        recv_size = 0
        print('content length', content_length)
        
        while recv_size < content_length:
            content += client.recv(1024)
            recv_size = len(content)
        # print(content.decode('utf-8'))
        
        # verify the data is correct
        if dic['MD5'] != hashlib.new('md5', content).hexdigest():
            print("ERROR: MD5 verification failed!")
    except Exception as e:
        print("get data error", e)
        
    return content
    
def message_handle(client, info):
    """
    Handle trained model return from clients
    """
    try:
        # receive trained model 
        content = get_data(client)
        parameter = json.loads(content.decode('utf-8'))
        
        # record model
        global trained_model
        if info not in trained_model:
            for k,v in parameter.items():
                parameter[k] = torch.Tensor(v)
            trained_model[info] = parameter
        else:
            print("Model has been received. Something wrong!")
    except Exception as e:
        print(e)
    
class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
    
def get_mnist_test(data_path: str = "./data"):
    # This function get clients local private data from ./data folder
    testset = torch.load(data_path+'/test_data_server.pt')

    return testset
        


if __name__ == '__main__':
    init()
    # connect to clients resource 
    accept_client()
    '''
    # start distributed machine learning
    # assume the data is distributed by server
    train_clients, val_clients, testloader = prepare_dataset(
        num_resource, val_ratio = 0.1
    )
    # distirbute data
    print("send data...")
    for i, key in enumerate(g_conn_pool):
       client = g_conn_pool[key]
       send_data(client, 'TRAIN_DATA', data = train_clients[i])
       send_data(client, 'VAL_DATA', data = val_clients[i])
    '''
    # start distributed machine learning
    # initial model
    model = Net(num_classes=10)
    # get dataset and construct a dataloaders
    testset = get_mnist_test()
    testloader = DataLoader(testset, batch_size=128)

    for i in range(epochs):
        # clean the history of recording table
        trained_model= {}
        # send initial model 
        
        # here implement sending model weight, not whole architecture
        # the shortage is have to know the arch first
        # TODO: later on modify ENcodeTensor to send the whole modle
        jsonstr = json.dumps(model.state_dict(), cls=EncodeTensor)
        # print(jsonstr)
        for k,v in g_conn_pool.items():
            print("send to ", k)
            send_data(v, "model", jsonstr.encode('utf-8'))
            # initial a new thread for each client waiting trained model
            thread = Thread(target=message_handle, args=(v, k), daemon=True)
            thread.start()
            thread.join()
        '''
            thread_pool.append(thread)
        for i in thread_pool:
            i.start()
        for i in thread_pool:
            i.join()
        '''

        print("Start merge the models from clients")
        merged_model = {}
        total_data_size = sum([ i['data_size'] for i in metaInfo_clients.values()])
        
        # g_conn_pool.keys()[0] ==> arbitrary clients id
        # trained_model[g_conn_pool.keys()[0]].keys() ==> names of layers parameters
        for para_name, weight in trained_model[list(g_conn_pool.keys())[0]].items():
            # initial meege model
            merged_model[para_name] = torch.zeros_like(weight)
            # strategy: FedAvg
            # iterate all clients' trained model
            for info in g_conn_pool.keys():
                # when num_resource=1, i.e. there is only one client, the parameters' 
                # MD5 received in this epochs should same as parameters' MD5 send out 
                # in next epochs 
                # because metaInfo_clients[info]['data_size']/total_data_size = 1.0.  
                # ********NOTE!!******** we should make sure compute 
                # metaInfo_clients[info]['data_size']/total_data_size first, 
                # otherwise model.state_dict() MD5 will change, because firstly compute
                # trained_model[info][para_name] * metaInfo_clients[info]['data_size']
                # will cause Floating-point numbers precision loss
                merged_model[para_name] += (trained_model[info][para_name] * (metaInfo_clients[info]['data_size']/total_data_size))
        
        # merge finish!
        model.load_state_dict(merged_model)
        print("merge success!")
        
        # merging is completed, then evaluate model on the test set
        loss, accuracy = test(model, testloader)
        print(f"{loss = }")
        print(f"{accuracy = }")
        
