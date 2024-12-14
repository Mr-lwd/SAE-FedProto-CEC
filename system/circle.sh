# nohup python -u main.py -algo FedSAE -data FashionMNIST_dir_0.3_imbalance_40 -gr 400 -nc 40 -lr 0.06 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_3/SAE_addTGP_gamma_1.0_buffer=0.5_joinrate_1.0_test.out 2>&1 &

# nohup python -u main.py -algo FedSAE -data FashionMNIST_dir_0.3_imbalance_40 -gr 400 -nc 40 -lr 0.06 -addTGP 0 > ./logs/FashionMNIST/FashionMNIST_dir_0.3_imbalance_40/epochs_3/SAE_gamma_1.0_buffer=0.5_joinrate_1.0.out 2>&1 &
