import socket
import json
from threading import Thread
import struct

ADDRESS = ('10.152.0.72', 9090)
client_type ='Distributed Machine Learning Client'

def send_data(client, cmd, **kv):
    global client_type
    jd = {}
    jd['COMMAND'] = cmd
    jd['client_type'] = client_type
    jd['data'] = kv

    jsonstr = json.dumps(jd)
    print('send: ' + jsonstr)
    client.sendall(jsonstr.encode('utf8'))

def message_handle(client):
    """
    Handle message for large scale data
    """
    print("heelo")
    while True:
        try:
            print("meta length")
            # receive meta length
            head = client.recv(4)
            '''
            dic_length = struct.unpack('i', head)[0]

            print("meta info")
            # receive meta info
            head_dic = client.recv(int(dic_length))
            dic = json.loads(head_dic.decode('utf-8'))

            # receive true data
            content_length = dic['size']
            print(content_length)
            content = b''
            recv_size = 0
            while recv_size < content_length:
                content += client.recv(1024)
                recv_size += len(content)

            data = content.decode(encoding='utf8')
            print(data)
            
            jd = json.loads(msg)
            cmd = jd['COMMAND']
            client_type = jd['client_type']
            if cmd == "TRAIN_DATA":
                print("train")
                send_data(client, 'SEND_DATA', data='get train')
            elif cmd == "VAL_DATA":
                print("test")
                send_data(client, 'SEND_DATA', data='get train')
            elif cmd == "MODEL":
                print("model")
            '''
        except Exception as e:
            print(e)
            break


if '__main__' == __name__:
    client = socket.socket()
    client.connect(ADDRESS)
    # get initial information from server
    msg = client.recv(1024).decode(encoding='utf8')
    jd = json.loads(msg)
    # update client name
    client_type += str(jd['id'])
    print(jd['connect_status'])
    send_data(client, 'CONNECT')
    print(client.recv(1024).decode(encoding='utf8'))

    # initial a new thread for each client
    thread = Thread(target=message_handle, args=client)
    # set to Daemon Thread
    # thread.setDaemon(True)
    thread.start()

