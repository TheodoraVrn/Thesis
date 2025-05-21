In different terminals:
* python SFLv1-server.py --port 8080 --num_clients 2 --epochs 2
* python SFLv1-client.py --client_id 0 --num_clients 2 --epochs 2
* python SFLv1-client.py --client_id 1 --num_clients 2 --epochs 2

Output:
* [SERVER] Listening on port 8080  
timmer start!  
[SERVER] Client 0 connected  
[SERVER] Client 0 training done  
[SERVER] Client 1 connected  
[SERVER] Client 1 training done  
[SERVER] FedAvg complete for epoch 1  
[SERVER] Client 0 connected  
[SERVER] Client 0 training done  
[SERVER] Client 1 connected  
[SERVER] Client 1 training done  
[SERVER] FedAvg complete for epoch 2  
TrainingTime: 404.15315794944763 sec  
* Files already downloaded and verified  
timmer start!  
[CLIENT 0] Epoch 1 complete  
[CLIENT 0] Epoch 2 complete  
[CLIENT 0] Finished  
TrainingTime: 277.593457698822 sec
* Files already downloaded and verified  
timmer start!  
[CLIENT 1] Epoch 1 complete  
[CLIENT 1] Epoch 2 complete  
[CLIENT 1] Finished  
TrainingTime: 392.2919178009033 sec
