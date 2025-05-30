# SFLv1-client.py (FedAvg + Split Learning, with bug fixes)

import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import pickle
import struct
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
import time
import random
from utils import datasets, dataset_settings


# --------- Socket helpers ---------
def send_msg(sock, msg):
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    msg_data = recvall(sock, msglen)
    if msg_data is None:
        return None
    return pickle.loads(msg_data)

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# --------- Dataset partitioning ---------
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label

# --------- Client model ---------
class ClientNet(nn.Module):
    def __init__(self):
        super(ClientNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return self.layer1(out)

# --------- Main ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("-c", "--num_clients", type=int, default=5, metavar="C", help="Number of Clients")
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    #parser.add_argument("-n", "--epochs", type=int, default=20, metavar="N", help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="Learning rate")
    parser.add_argument("--dataset", type=str, default="cifar10", help="States dataset to be used")
    parser.add_argument("-b", "--batch_size", type=int, default=32, metavar="B", help="Batch size")
    parser.add_argument("--test_batch_size", type=int, default=512, metavar="B", help="Batch size")
    parser.add_argument("--datapoints", type=int, default=500, help='Number of data samples per client in setting 1')
    parser.add_argument("--setting", type=str, default="setting1")
    args = parser.parse_args()

    # itinialize parameters from given arguments
    SEED = args.seed    ## Set Hyperparameters and Configuration
    num_users = args.num_clients
    #epochs = args.epochs     
    lr = args.lr
    dataset = args.dataset
    if args.dataset == "mnist" or args.dataset == "fmnist":
        input_channels = 1
    else:
        input_channels = 3

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    

    train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(dataset, "data", num_users, args.datapoints, False)
    #dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print("Dataset used: ", dataset)
    print("full train set: ", len(train_full_dataset))
    print("full test set: ", len(test_full_dataset))
    dict_users = dataset_iid(train_full_dataset, args.num_clients)
    dict_users_test = dataset_iid(test_full_dataset, args.num_clients)
    train_loader = DataLoader(DatasetSplit(train_full_dataset, dict_users[args.client_id]), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(DatasetSplit(test_full_dataset, dict_users_test[args.client_id]), batch_size=args.test_batch_size, shuffle=False)

    model = ClientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sock = socket.socket()
    sock.connect((args.host, args.port))
    epochs = recv_msg(sock)   # get epoch
    print("Epoch received: ", epochs)
    total_batch = len(train_loader)
    msg = total_batch
    send_msg(sock, msg)   # send total_batch of train dataset


    start_time = time.time()    # store start time
    print("timmer start!")

    for epoch in range(epochs):
        #sock = socket.socket()
        #sock.connect((args.host, args.port))
        print(f"[CLIENT {args.client_id}] Epoch {epoch+1} training starting")
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            fx = model(images)

            # Clone and detach for sending
            fx_send = fx.detach().clone().requires_grad_()

            send_msg(sock, {
                'client_id': args.client_id,
                'epoch': epoch,
                'activations': fx_send.cpu(),
                'labels': labels.cpu(),
                'eval': False
            })

            dfx = recv_msg(sock)
            if dfx is None:
                print("[CLIENT] Server disconnected unexpectedly.")
                break

            fx.backward(gradient=dfx.to(device))
            optimizer.step()
        '''
        #============ Per-client Evaluation ==================
        model.eval()
        with torch.no_grad():
            len_batch = len(test_loader)
            #for batch_idx, (images, labels) in enumerate(test_loader):
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                #---------forward prop-------------
                fx = model(images)
                
                # Sending activations to server 
                #evaluate_server(fx, labels, self.idx, len_batch, ell)

                send_msg(sock, {
                    'client_id': args.client_id,
                    'epoch': epoch,
                    'activations': fx.detach().cpu(),
                    'labels': labels.cpu(),
                    'eval': True
                })'''

        ##
        print(f"[CLIENT {args.client_id}] Epoch {epoch+1} complete")

        #sock.close()

        
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
               

    print(f"[CLIENT {args.client_id}] Finished")
end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))
