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

def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Naive Split leaning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "-c",
        "--number_of_clients",
        type=int,
        default=5,
        metavar="C",
        help="Number of Clients",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="Total number of epochs to train",
    )

    parser.add_argument(
        "--fac",
        type=float,
        default= 1.0,
        metavar="N",
        help="fraction of active/participating clients, if 1 then all clients participate in SFLV1",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="States dataset to be used",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        metavar="B",
        help="Batch size",
    )

    parser.add_argument(

        "--test_batch_size",
        type=int,
        default=512,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="setting1",
     
    )

    parser.add_argument(
        "--datapoints",
        type=int,
        default=500,
        help='NUmber of data samples per client in setting 1'
     
    )

    parser.add_argument(
        "--opt_iden",
        type=str,
        default="",
        help="optional identifier of experiment",
    )

    args = parser.parse_args()
    return args

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
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    global_models = [ServerNet().to(device) for _ in range(args.num_clients)]

    s = socket.socket()
    s.bind(('0.0.0.0', args.port))
    s.listen(1)
    print(f"[SERVER] Listening on port {args.port}")

    start_time = time.time()    # store start time
    print("timmer start!")

    for epoch in range(args.epochs):
        local_states = []

        for client_id in range(args.num_clients):
            conn, addr = s.accept()
            print(f"[SERVER] Client {client_id} connected")

            model = global_models[client_id]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            while True:
                data = recv_msg(conn)
                if data is None:
                    break
                if data['epoch'] > epoch:
                    break

                fx = data['activations'].to(device).requires_grad_()
                labels = data['labels'].to(device)

                optimizer.zero_grad()
                out = model(fx)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                send_msg(conn, fx.grad.cpu())

            conn.close()
            local_states.append(copy.deepcopy(model.state_dict()))
            print(f"[SERVER] Client {client_id} training done")

        global_state = FedAvg(local_states)
        for i in range(args.num_clients):
            global_models[i].load_state_dict(global_state)

        print(f"[SERVER] FedAvg complete for epoch {epoch+1}")
end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))
