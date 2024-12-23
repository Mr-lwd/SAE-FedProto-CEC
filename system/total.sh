nohup python -u main.py -algo FedSAE -gamma 1 -test_useglclassifier 1 -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -lr 0.06 -momentum 0.5 -ls 10 -lam 1 -m HtFE8 -drawtsne 1 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/lr_006_mo_0.5_lam_1_batch_256_FedSAE_gam_1.out 2>&1 &

# nohup python -u main.py -algo FedProto -addTGP 0 -data Cifar10_dir_0.3_imbalance_40 -gr 200 -lbs 256 -nc 40 -lr 0.0005 -ls 10 -lam 1 -m HtFE8 -drawtsne 0 -drawround 10 > ./logs/Cifar10/Cifar10_dir_0.3_imbalance_40/epochs_10/RMSprop_lr_5e-4_lam_1_batch_256_FedProto.out 2>&1 &
