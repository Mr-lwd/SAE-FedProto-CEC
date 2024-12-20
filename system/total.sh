
nohup python -u main.py -algo FedSAE -addTGP 0 -gamma 1.0 -data FashionMNIST_dir_0.3_imbalance_40 -gr 300 -nc 40 -lr 0.03 -ls 10 -lam 2 -test_useglclassifier 1 -SAEbeta 1 -m HCNNs8 -mart 1000 -mixclassifier 0 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/fedSAE_1220test.out 2>&1 &

# nohup python -u main.py -algo FedSAE -addTGP 0 -gamma 1.0 -data MNIST_dir_0.3_imbalance_40 -gr 200 -nc 40 -lr 0.03 -ls 10 -lam 2 -test_useglclassifier 1 -SAEbeta 1 -m HCNNs8 -mart 1000 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/lamda_2_FedSAE_noTGP_gl_use_clients_gamma_1.0_buffer=1.0.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data FashionMNIST_dir_0.3_imbalance_40 -gr 200 -nc 40 -lr 0.03 -ls 10 -lam 10 -m HCNNs8 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/FedProto.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -nc 40 -lr 0.03 -ls 10 -lam 10 -m HtFE8 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/FedProto.out 2>&1 && \

# nohup python -u main.py -algo FedSAE -addTGP 0 -gamma 1.0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -nc 40 -lr 0.03 -ls 10 -lam 10 -test_useglclassifier 1 -m HtFE8 -mart 100 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lamda_10_FedSAE_noTGP_gl_use_clients_gamma_1.0_buffer=1.0.out 2>&1 &

# nohup python -u main.py -algo FedSAE -addTGP 0 -gamma 0.7 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -nc 40 -lr 0.03 -ls 10 -lam 2 -test_useglclassifier 0 -m HtFE8 -mart 100 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lam_2_FedSAE_noTGP_gl_use_clients_gamma_0.7_buffer=1.0.out 2>&1 &

# nohup python -u main.py -algo FedTGP -data FashionMNIST_dir_0.3_imbalance_40 -gr 300 -nc 40 -lr 0.03 -ls 10 -lam 10 -m HCNNs8 -tam 1 -addmse 1 -mart 1000 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/lamda_10_FedTGP_addMSE.out 2>&1 &

# nohup python -u main.py -algo FedTGP -data Cifar10_dir_0.3_imbalance_40 -gr 300 -nc 40 -lr 0.03 -ls 10 -lam 2 -m HtFE8 -tam 0 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lamda_2_FedTGP.out 2>&1 &

# nohup python -u main.py -algo FedSAE -addTGP 1 -gamma 1.0 -data FashionMNIST_dir_0.3_imbalance_40 -gr 300 -nc 40 -lr 0.03 -ls 10 -lam 19 -test_useglclassifier 1 -SAEbeta 1 -m HCNNs8 -mart 1000 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/lamda_5_SAE_addTGP_gl_use_clients_gamma_1.0_beta_1.0_buffer=1.0.out 2>&1 && \

# nohup python -u main.py -algo FedSAE -addTGP 1 -gamma 1.0 -data MNIST_dir_0.3_imbalance_40 -gr 300 -nc 40 -lr 0.03 -ls 10 -lam 10 -test_useglclassifier 1 -SAEbeta 1 -m HCNNs8 -mart 1000 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/lamda_5_SAE_addTGP_gl_use_clients_gamma_1.0_beta_1.0_buffer=1.0.out 2>&1 &

# nohup python -u main.py -algo FedSAE -addTGP 1 -gamma 1.0 -data Cifar10_dir_0.3_imbalance_40 -gr 300 -nc 40 -lr 0.03 -ls 10 -lam 10 -test_useglclassifier 1 -SAEbeta 1 -m HtFE8 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lamda_10_SAE_addTGP_gl_use_clients_gamma_1.0_beta_1.0_buffer=1.0.out 2>&1 && \
