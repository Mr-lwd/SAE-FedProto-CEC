# fedTGP
nohup python -u main.py -algo FedTGP -data FashionMNIST_dir_0.3_imbalance_40 -gr 100 -nc 40 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_3/fedTGP.out 2>&1 && \
nohup python -u main.py -algo FedTGP -data MNIST_dir_0.3_imbalance_40 -gr 100 -nc 40 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_3/fedTGP.out 2>&1 && \

nohup python -u main.py -algo FedProto -data MNIST_dir_0.3_imbalance_40 -gr 100 -nc 40 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_3/fedproto.out 2>&1 && \
nohup python -u main.py -algo FedSAE -data FashionMNIST_dir_0.3_imbalance_40 -gr 200 -nc 40 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_3/SAE_gamma_1.0_buffer=0.5_joinrate_1.0.out 2>&1 &

