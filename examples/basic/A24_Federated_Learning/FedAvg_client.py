import socket
import json
from threading import Thread
import struct
import time
import hashlib

import torch
from torch.utils.data import Dataset

from json import JSONEncoder

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

ADDRESS = ('10.152.0.72', 9090)
client_type ='Federated Machine Learning Client'

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NpEncoder, self).default(obj)

def get_local_mnist(data_path: str = "./data"):
    # This function get clients local private data from ./data folder
    
    trainset = torch.load(data_path+'/train_data_client.pt')
    testset = torch.load(data_path+'/test_data_client.pt')

    return trainset, testset

def send_init_data(client, data_size):
    global client_type
    jd = {}
    jd['client_type'] = client_type
    jd['data_size'] = data_size

    jsonstr = json.dumps(jd)
    print('send: ' + jsonstr)
    client.sendall(jsonstr.encode('utf8'))
    
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
            return None
    except Exception as e:
        print("get data error", e)
        
    return content
    
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

def federated_learning(client):
    """
    Handle message for large scale data
    """
    while True:
        try:
            # receive initial model from server
            content = get_data(client)
            if content == None:
                print("federated learning terminate!")
                break
            
            # generate the model
            parameter = json.loads(content.decode('utf-8'))
            # all tensor become list in json content
            # TODO: better way is use JSONDECODER to do it 
            
            # convert it back to tensor
            for k,v in parameter.items():
                parameter[k] = torch.Tensor(v)
                
            model = Net(num_classes=10)
            model.load_state_dict(parameter)
            
            # train on own private data
            model = run_centralised(model, lr=0.01)
            
            # return trained model back to server
            jsonstr = json.dumps(model.state_dict(), cls=EncodeTensor)
            send_data(client, "trained_model", jsonstr.encode('utf-8'))
            # print(model.state_dict())
        except Exception as e:
            print(e)
            break
            
def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net


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
    
def run_centralised(model, lr: float, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # initial the model, modek comes from server as parameter

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # get dataset and construct a dataloaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128)

    # train for the specified number of epochs
    trained_model = train(model, trainloader, optim, 1)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, testloader)
    print(f"{loss = }")
    print(f"{accuracy = }")
    
    # input model is a hardcopy, not an address, need to return trained_model  
    return model


if '__main__' == __name__:
    # prepare the data
    trainset, testset = get_local_mnist()
    print("loading data... success!")
    
    # connect to server
    client = socket.socket()
    client.connect(ADDRESS)
    
    # get initial information from server
    msg = client.recv(1024).decode(encoding='utf8')
    jd = json.loads(msg)
    # update client name
    client_type += str(jd['id'])
    print(jd['connect_status'])
    send_init_data(client, len(trainset))
    
    federated_learning(client)
