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
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dict_users = dataset_iid(dataset, args.num_clients)
    loader = DataLoader(DatasetSplit(dataset, dict_users[args.client_id]), batch_size=args.batch_size, shuffle=True)

    model = ClientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #sock = socket.socket()
    #sock.connect((args.host, args.port))

    start_time = time.time()    # store start time
    print("timmer start!")

    for epoch in range(args.epochs):
        sock = socket.socket()
        sock.connect((args.host, args.port))
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            fx = model(images)

            # Clone and detach for sending
            fx_send = fx.detach().clone().requires_grad_()

            send_msg(sock, {
                'client_id': args.client_id,
                'epoch': epoch,
                'activations': fx_send.cpu(),
                'labels': labels.cpu()
            })

            dfx = recv_msg(sock)
            if dfx is None:
                print("[CLIENT] Server disconnected unexpectedly.")
                break

            fx.backward(gradient=dfx.to(device))
            optimizer.step()

        print(f"[CLIENT {args.client_id}] Epoch {epoch+1} complete")

        sock.close()
    print(f"[CLIENT {args.client_id}] Finished")
end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))
