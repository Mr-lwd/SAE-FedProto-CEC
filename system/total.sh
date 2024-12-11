# nohup python -u main.py > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_3/fedproto.out 2>&1 &
# nohup python -u main.py -algo FedSAE -dataset FashionMNIST_dir_0.3_imbalance_40 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_3/SAE_gamma_1.0_buffer=0.5_joinrate_1.0.out 2>&1 


# nohup python -u main.py > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_3/fedproto_lr006.out 2>&1 &
# nohup python -u main.py > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_3/SAE_gamma_1.0_buffer=0.5_joinrate_1.0.out 2>&1 &


# nohup python -u main.py -algo FedProto -dataset MNIST_dir_0.3_imbalance_40 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_3/fedproto.out 2>&1 
# nohup python -u main.py > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_3/SAE_gamma_1.0_buffer=0.5_joinrate_1.0.out 2>&1 &
