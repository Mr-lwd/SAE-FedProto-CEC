# nohup python -u main.py -algo FedSAE -data FashionMNIST_dir_0.3_imbalance_40 -gr 150 -nc 40 -lr 0.06 -ls 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/SAE_noTGP_gamma_1.0_buffer=1.0_joinrate_1.0.out 2>&1 && \

nohup python -u main.py -algo FedProto -data FashionMNIST_dir_0.3_imbalance_40 -gr 150 -nc 40 -lr 0.06 -ls 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/fedproto.out 2>&1 &

# nohup python -u main.py -algo FedTGP -data MNIST_dir_0.3_imbalance_40 -gr 100 -nc 40 -lr 0.06 -ls 10 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/FedTGP_avg.out 2>&1 &

# nohup python -u main.py -algo FedSAE -gamma 0.5 -data MNIST_dir_0.3_imbalance_40 -gr 100 -nc 40 -lr 0.06 -ls 10 -test_useglclassifier 0 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/SAE_addTGP_gamma_0.5_buffer=1.0_joinrate_1.0.out 2>&1 &


# nohup python -u main.py -algo FedTGP -gamma 1.0 -data FashionMNIST_dir_0.3_imbalance_40 -gr 200 -nc 40 -lr 0.06 -ls 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/FedTGP.out 2>&1 &