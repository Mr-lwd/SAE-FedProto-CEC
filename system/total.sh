# nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data Cifar10_dir_0.3_imbalance_40  -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.08 -momentum 0.5 -lam 2  -ls 10  -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lr_008_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out 2>&1 &

# nohup python -u main.py -algo FedTGP -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -lr 0.06 -momentum 0.5 -ls 10 -lam 1 -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lr_006_mo_0.5_lam_1_batch_256_FedTGP_error_collectprotos.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.08 -momentum 0.7 -lam 2  -ls 10 -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/SGD_wd_lr_008_lam_2_momentum_0.7_batch_256_FedProto.out 2>&1 &

# nohup python -u main.py -algo FedTGP -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -lr 0.08 -momentum 0.5 -ls 10 -lam 1 -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/Adam_lr_008_mo_0.5_lam_1_batch_256_FedTGP.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -lr -ls 10 -lam 1 -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/RMSprop_lr_5e-4_lam_1_batch_256_FedProto.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data MNIST_dir_0.3_imbalance_40 -gr 150 -lbs 256 -nc 40 -optimizer SGD -lr 0.06  -momentum 0.5 -lam 2 -ls 10 -m HCNNs8 -drawtsne 1 -drawround 10 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/SGD_lr_006_wd_1e-5_mo_0.5_lam_2_batch_256_FedProto.out 2>&1 && \


# nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data MNIST_dir_0.3_imbalance_40  -gr 150 -lbs 256 -nc 40 -optimizer SGD -lr 0.06 -momentum 0.5 -lam 2  -ls 10  -m HCNNs8 -drawtsne 1 -drawround 10 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/lr_006_wd_1e-5_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out 2>&1 && \

# nohup python -u main.py -algo FedProto -addTGP 0 -data FashionMNIST_dir_0.3_imbalance_40 -gr 150 -lbs 256 -nc 40 -optimizer SGD -lr 0.06  -momentum 0.5 -lam 2 -ls 10 -m HCNNs8 -drawtsne 1 -drawround 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/SGD_lr_006_wd_1e-5_mo_0.5_lam_2_batch_256_FedProto.out 2>&1 && \


# nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data FashionMNIST_dir_0.3_imbalance_40  -gr 150 -lbs 256 -nc 40 -optimizer SGD -lr 0.06 -momentum 0.5 -lam 2  -ls 10 -m HCNNs8 -drawtsne 1 -drawround 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/lr_006_wd_1e-5_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out 2>&1 &

# nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data Cifar10_dir_0.3_imbalance_40  -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.06 -momentum 0.5 -lam 2  -ls 10  -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lr_006_wd_1e-5_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data MNIST_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.08  -momentum 0.5 -lam 2 -ls 10 -m HCNNs8 -drawtsne 1 -drawround 10 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/SGD_lr_008_lam_2_momentum_0.5_batch_256_FedProto.out 2>&1 && \

# nohup python -u main.py -algo FedProto -addTGP 0 -data FashionMNIST_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.08  -momentum 0.5 -lam 2 -ls 10 -m HCNNs8 -drawtsne 1 -drawround 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/SGD_lr_008_lam_2_momentum_0.5_batch_256_FedProto.out 2>&1 && \

# nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data MNIST_dir_0.3_imbalance_40  -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.08 -momentum 0.5 -lam 2  -ls 10  -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/lr_008_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out 2>&1 && \

# nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data FashionMNIST_dir_0.3_imbalance_40  -gr 200 -lbs 256 -nc 40 -optimizer SGD -lr 0.08 -momentum 0.5 -lam 2  -ls 10 -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_10/lr_008_mo_0.5_lam_2_batch_256_FedSAE_gam_1.out 2>&1 &


nohup python -u main.py -algo FedProto -addTGP 0 -dev cpu -data MNIST_dir_0.3_imbalance_40 -gr 10 -lbs 256 -nc 40 -optimizer SGD -lr 0.06  -momentum 0.5 -lam 2 -ls 10 -m HCNNs8 -drawtsne 0 -drawround 10 > ./logs/MNIST/MNIST_dir_0.3_imbalance_40/epochs_10/SGD_lr_006_wd_1e-5_mo_0.5_lam_2_batch_256_FedProto_hardware.out 2>&1 &
