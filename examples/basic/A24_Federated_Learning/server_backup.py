import socket
from threading import Thread
import time
import json
import hashlib
import struct

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import random_split

ADDRESS = ('0.0.0.0', 9090)

g_socket_server = None # listening socket

g_conn_pool = {}  # connect pool

num_actived_clients = 0
num_resource = 1
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
        client, info = g_socket_server.accept()  # block for client connect
        print("connect success")
        '''
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
        cmd = jd['COMMAND']
        client_type = jd['client_type']

        if 'CONNECT' == cmd:
            g_conn_pool[client_type] = client
            print('on client connect: ' + client_type, info)
            num_actived_clients += 1
        '''
        g_conn_pool['test'] = client
        num_actived_clients += 1
        # initial a new thread for each client
        thread = Thread(target=message_handle, args=(client, info))
        # set to Daemon Thread
        thread.setDaemon(True)
        thread.start()

def message_handle(client, info):
    """
    Handle message
    """
    while True:
        try:
            bytes = client.recv(1024)
            msg = bytes.decode(encoding='utf8')
            jd = json.loads(msg)
            cmd = jd['COMMAND']
            client_type = jd['client_type']
            if 'MODEL' == cmd:
                print('on client connect: ' + client_type, info)
            elif 'SEND_DATA' == cmd:
                print('recv client msg: ' + client_type, jd['data'])
        except Exception as e:
            print(e)
            remove_client(client_type)
            break

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

def send_data(client, cmd, data):
    """
    Send message for large scale data avoiding Stick Package
    """
    # construct meta info
    total_data_bytes = len(data)
    print(total_data_bytes)
    head_dic = {
        'time': time.localtime(time.time()),
        'size': total_data_bytes,
        'MD5': 'XXXXXXXXXXX', # hashlib.md5(data),
        'data_type': cmd
    }
    head_dic_bytes = json.dumps(head_dic).encode('utf-8')
    print(len(head_dic_bytes))
    # compute meta data length and send info in 4 bytes
    head = struct.pack('i', len(head_dic_bytes))

    # send meta length
    client.send(head)

    # send meta info
    client.send(head_dic_bytes)

    # send true data
    jsonstr = json.dumps(data)
    client.send(jsonstr.encode('utf-8'))

if __name__ == '__main__':
    init()
    # connect to clients resource 
    accept_client()

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
    '''
    send_data(client, 'VAL_DATA', data = val_clients[i])

    epochs = 5
    #for i in range(epochs):

    for k,v in g_conn_pool.items():
        print("send to ", k)
        msg = "hello" + str(c)
        c+=1
        v.sendall(msg.encode('utf8'))
    '''
    while True:
        time.sleep(0.1)
