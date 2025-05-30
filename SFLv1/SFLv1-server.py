# SFLv1-server.py (FedAvg + Split Learning, with robust recv fix)

import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import pickle
import struct
import argparse
import copy
import time
import random
import numpy as np
from tqdm import tqdm


#=============================================================================
#                         Socket functions
#============================================================================= 
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

# --------- Server model ---------
class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        out = self.layer2(x)
        return self.fc(out.view(out.size(0), -1))

# --------- FedAvg ---------
def FedAvg(models):
    avg = copy.deepcopy(models[0])
    for k in avg:
        for i in range(1, len(models)):
            avg[k] += models[i][k]
        avg[k] = avg[k] / len(models)
    return avg

# --------- Main ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    #parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument("-c", "--num_clients", type=int, default=5, metavar="C", help="Number of Clients")
    #parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument("-n", "--epochs", type=int, default=20, metavar="N", help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="Learning rate")
    parser.add_argument("--dataset", type=str, default="mnist", help="States dataset to be used")
    # batch size received from client
    # test batch size received from client
    args = parser.parse_args()

    SEED = args.seed    ## Set Hyperparameters and Configuration
    num_users = args.num_clients
    epochs = args.epochs     
    lr = args.lr
    DATASET = args.dataset
    if args.dataset == "mnist" or args.dataset == "fmnist":
        input_channels = 1
    else:
        input_channels = 3

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    global_models = [ServerNet().to(device) for _ in range(num_users)]

    s = socket.socket()
    s.bind(('0.0.0.0', args.port))
    s.listen(1)
    print(f"[SERVER] Listening on port {args.port}")

    clientsoclist = []      # socket connections with clients

    for i in range(num_users):
        conn, addr = s.accept()
        print(f"[SERVER] Client {i} connected")
        clientsoclist.append(conn)    # append client socket on list
        send_msg(conn, epochs)    #send epoch
        total_batch = recv_msg(conn)   # get total_batch of train dataset


    start_time = time.time()    # store start time
    print("timmer start!")

    count_while = 0
    
    for epoch in range(epochs):
        local_states = []

        for client_id in range(num_users):
            #conn, addr = s.accept()
            #print(f"[SERVER] Client {client_id} connected")

            model = global_models[client_id]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            #while True:
            for i in tqdm(range(total_batch), ncols=100, desc='Epoch {} Client{} '.format(epoch+1, client_id)):
                data = recv_msg(clientsoclist[client_id])  ## receive total batch from client
                if data is None:
                    break
                '''
                if data['eval']== True:
                    with torch.no_grad():
                        fx = data['activations'].to(device)
                        labels = data['labels'].to(device)
                        #---------forward prop-------------
                        outputs = model(fx)
                        # calculate loss
                        loss = criterion(outputs, labels)
                        # calculate accuracy
                        preds = outputs.max(1, keepdim=True)[1]
                        correct = preds.eq(labels.view_as(preds)).sum()
                        acc = 100.00 *correct.float()/preds.shape[0]

                        #preds = outputs.argmax(dim=1).cpu().tolist()
                        #send_msg(conn, preds)
                        print(f"[CLIENT {data['client_id']}] Test Accuracy after Epoch {epoch+1}: {acc:.2f}%")
                    continue
                '''
                if data['epoch'] > epoch:
                    break

                fx = data['activations'].to(device).requires_grad_()
                labels = data['labels'].to(device)

                optimizer.zero_grad()
                out = model(fx)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                send_msg(clientsoclist[client_id], fx.grad.cpu())
            count_while = count_while+1

            #conn.close()
            local_states.append(copy.deepcopy(model.state_dict()))
            print(f"[SERVER] Client {client_id} training done")
            '''
            test_data = recv_msg(clientsoclist[client_id])
            if data['eval']== True:
                with torch.no_grad():
                    fx = data['activations'].to(device)
                    labels = data['labels'].to(device)
                    #---------forward prop-------------
                    outputs = model(fx)
                    # calculate loss
                    loss = criterion(outputs, labels)
                    # calculate accuracy
                    preds = outputs.max(1, keepdim=True)[1]
                    correct = preds.eq(labels.view_as(preds)).sum()
                    acc = 100.00 *correct.float()/preds.shape[0]

                    #preds = outputs.argmax(dim=1).cpu().tolist()
                    #send_msg(conn, preds)
                    print(f"[CLIENT {data['client_id'].to(device)}] Test Accuracy after Epoch {epoch+1}: {acc:.2f}%")


        # Accept eval messages from each client after training
        for client_id in range(args.num_clients):
            model = global_models[client_id]
            data = recv_msg(conn)
            if data is None:
                break
            if data['eval']==True:
                with torch.no_grad():
                    fx = data['activations'].to(device)
                    labels = data['labels'].to(device)
                    #---------forward prop-------------
                    outputs = model(fx)
                    # calculate loss
                    loss = criterion(outputs, labels)
                    # calculate accuracy
                    preds = outputs.max(1, keepdim=True)[1]
                    correct = preds.eq(labels.view_as(preds)).sum()
                    acc = 100.00 *correct.float()/preds.shape[0]
                    print(f"[CLIENT {client_id}] Test Accuracy after Epoch {epoch+1}: {acc:.2f}%")
                        
                        #split = data.get('split', 'unknown').upper()
                        #print(f"[CLIENT {client_id}] {split} Accuracy after Epoch {epoch+1}: {acc:.2f}%")
            #conn.close()
            '''
        global_state = FedAvg(local_states)
        for i in range(num_users):
            global_models[i].load_state_dict(global_state)

        print(f"[SERVER] FedAvg complete for epoch {epoch+1}")


end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))