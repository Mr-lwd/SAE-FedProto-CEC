#!/usr/bin/env python

import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
import random
from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from flcore.servers.servertgp import FedTGP
from flcore.servers.serverSAE import FedSAE
from flcore.servers.servergen import FedGen
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.servergh import FedGH
from flcore.servers.serverktl_stylegan_xl import FedKTL as FedKTL_stylegan_xl
from flcore.servers.serverktl_stylegan_3 import FedKTL as FedKTL_stylegan_3
from flcore.servers.serverktl_stable_diffusion import FedKTL as FedKTL_stable_diffusion

from utils.result_utils import average_data
from utils.mem_utils import MemReporter


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.models
        if args.model_family == "HtFE2":
            args.models = [
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
            ]

        elif args.model_family == "HtFE3":
            args.models = [
                "resnet10(num_classes=args.num_classes)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
            ]

        elif args.model_family == "HtFE4":
            args.models = [
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
                "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
                "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
            ]

        elif args.model_family == "HtFE8":
            args.models = [
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
                # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)',
                "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
                "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)",
            ]

        elif args.model_family == "HtFE9":
            args.models = [
                "resnet4(num_classes=args.num_classes)",
                "resnet6(num_classes=args.num_classes)",
                "resnet8(num_classes=args.num_classes)",
                "resnet10(num_classes=args.num_classes)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)",
            ]

        elif args.model_family == "HtFE8-HtC4":
            args.models = [
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
                "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
                "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)",
            ]
            args.global_model = (
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)"
            )
            args.heads = [
                "Head(hidden_dims=[512], num_classes=args.num_classes)",
                "Head(hidden_dims=[512, 512], num_classes=args.num_classes)",
                "Head(hidden_dims=[512, 256], num_classes=args.num_classes)",
                "Head(hidden_dims=[512, 128], num_classes=args.num_classes)",
            ]

        elif args.model_family == "Res34-HtC4":
            args.models = [
                "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
            ]
            args.global_model = (
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)"
            )
            args.heads = [
                "Head(hidden_dims=[512], num_classes=args.num_classes)",
                "Head(hidden_dims=[512, 512], num_classes=args.num_classes)",
                "Head(hidden_dims=[512, 256], num_classes=args.num_classes)",
                "Head(hidden_dims=[512, 128], num_classes=args.num_classes)",
            ]

        elif args.model_family == "HCNNs8":
            args.models = [
                "CNN(num_cov=1, hidden_dims=[], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=2, hidden_dims=[], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=1, hidden_dims=[512], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=2, hidden_dims=[512], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=1, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=2, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=1, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)",
                "CNN(num_cov=2, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)",
            ]

        elif args.model_family == "ViTs":
            args.models = [
                "torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)",
                "torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)",
                "torchvision.models.vit_l_16(image_size=32, num_classes=args.num_classes)",
                "torchvision.models.vit_l_32(image_size=32, num_classes=args.num_classes)",
            ]

        elif args.model_family == "HtM10":
            args.models = [
                "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
                "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
                "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)",
                "torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)",
                # "torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)",
                # "torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)",
            ]

        elif args.model_family == "NLP_all":
            args.models = [
                "fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)",
                "LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)",
                "BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)",
                "TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, num_classes=args.num_classes)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)",
            ]

        elif args.model_family == "NLP_Transformers-nhead=8":
            args.models = [
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)",
            ]

        elif args.model_family == "NLP_Transformers-nlayers=4":
            args.models = [
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
            ]

        elif args.model_family == "NLP_Transformers":
            args.models = [
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)",
                "TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)",
            ]

        elif args.model_family == "MLPs":
            args.models = [
                "AmazonMLP(feature_dim=[])",
                "AmazonMLP(feature_dim=[200])",
                "AmazonMLP(feature_dim=[500])",
                "AmazonMLP(feature_dim=[1000, 500])",
                "AmazonMLP(feature_dim=[1000, 500, 200])",
            ]

        elif args.model_family == "MLP_1layer":
            args.models = [
                "AmazonMLP(feature_dim=[200])",
                "AmazonMLP(feature_dim=[500])",
            ]

        elif args.model_family == "MLP_layers":
            args.models = [
                "AmazonMLP(feature_dim=[500])",
                "AmazonMLP(feature_dim=[1000, 500])",
                "AmazonMLP(feature_dim=[1000, 500, 200])",
            ]

        else:
            raise NotImplementedError

        for model in args.models:
            print(model)

        # select algorithm
        if args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "FedSAE":
            server = FedSAE(args, i)

        elif args.algorithm == "FedGen":
            server = FedGen(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        elif args.algorithm == "LG-FedAvg":
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            server = FedKD(args, i)

        elif args.algorithm == "FedGH":
            server = FedGH(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)

        elif args.algorithm == "FedKTL-stylegan-xl":
            server = FedKTL_stylegan_xl(args, i)

        elif args.algorithm == "FedKTL-stylegan-3":
            server = FedKTL_stylegan_3(args, i)

        elif args.algorithm == "FedKTL-stable-diffusion":
            server = FedKTL_stable_diffusion(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(
        dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times
    )

    print("All done!")

    reporter.report()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    total_start = time.time()
    num_edges = 10
    edge_ratio = 1.0
    buffer_size = int(num_edges * edge_ratio)

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "-go", "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument(
        "-data", "--dataset", type=str, default="MNIST_dir_0.3_imbalance_40"
    )
    # parser.add_argument(
    #     "-data", "--dataset", type=str, default="FashionMNIST_dir_0.3_imbalance_40"
    # )
    # parser.add_argument(
    #     "-data", "--dataset", type=str, default="Cifar10_dir_0.3_imbalance_40"
    # )
    parser.add_argument("-nb", "--num_classes", type=int, default=10)
    parser.add_argument("-m", "--model_family", type=str, default="HCNNs8")
    # parser.add_argument("-m", "--model_family", type=str, default="HtM10")
    parser.add_argument("-lbs", "--batch_size", type=int, default=256)
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.02,
        help="Local learning rate",
    )
    parser.add_argument("-usche", "--use_decay_scheduler", type=bool, default=False)
    parser.add_argument("-ld", "--learning_rate_decay", type=bool, default=False)
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument("-gr", "--global_rounds", type=int, default=100)
    parser.add_argument(
        "-edge_epochs", "--edge_epochs", type=int, default=1, help="edge epoches"
    )
    parser.add_argument(
        "-ls",
        "--local_epochs",
        type=int,
        default=10,
        help="Multiple update steps in one local epoch.",
    )
    # parser.add_argument("-algo", "--algorithm", type=str, default="FedProto")
    # parser.add_argument("-algo", "--algorithm", type=str, default="FedSAE")
    parser.add_argument("-algo", "--algorithm", type=str, default="FedTGP")
    parser.add_argument(
        "-jr",
        "--join_ratio",
        type=float,
        default=1.0,
        # default=0.8,
        help="Ratio of clients per round",
    )
    parser.add_argument(
        "-rjr",
        "--random_join_ratio",
        type=bool,
        default=False,
        help="Random ratio of clients per round",
    )
    parser.add_argument(
        "-nc", "--num_clients", type=int, default=40, help="Total number of clients"
    )
    parser.add_argument(
        "-ne", "--num_edges", type=int, default=num_edges, help="Total number of edges"
    )

    parser.add_argument(
        "-pv", "--prev", type=int, default=0, help="Previous Running times"
    )
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        "-eg", "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="temp")
    parser.add_argument("-ab", "--auto_break", type=bool, default=False)
    parser.add_argument("-fd", "--feature_dim", type=int, default=512)
    parser.add_argument("-vs", "--vocab_size", type=int, default=98635)
    parser.add_argument("-ml", "--max_len", type=int, default=200)
    # practical
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )
    parser.add_argument(
        "-tsr",
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when training locally",
    )
    parser.add_argument(
        "-ssr",
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when sending global model",
    )
    parser.add_argument(
        "-ts",
        "--time_select",
        type=bool,
        default=False,
        help="Whether to group and select clients at each round according to time cost",
    )
    parser.add_argument(
        "-tth",
        "--time_threthold",
        type=float,
        default=10000,
        help="The threthold for droping slow clients",
    )
    # FedProto
    parser.add_argument("-lam", "--lamda", type=float, default=10)
    parser.add_argument(
        "-trans_delay_simulate", "--trans_delay_simulate", type=bool, default=False
    )
    # agg_type 0--按数据量平均 1--按类平均原型聚合
    parser.add_argument("-agg_type", "--agg_type", type=int, default=0)

    # FedSAE
    parser.add_argument(
        "-bs", "--buffersize", type=int, default=buffer_size
    )  # 与边缘数量相等则等价于全同步
    parser.add_argument("-mixclassifier", "--mixclassifier", type=int, default=0)
    parser.add_argument(
        "-gl_use_clients", "--gl_use_clients", type=int, default=1
    )
    parser.add_argument(
        "-test_useglclassifier", "--test_useglclassifier", type=int, default=1
    )
    parser.add_argument(
        "-tam", "--tgpaddmse", type=int, default=0
    )
    parser.add_argument("-gamma", "--gamma", type=float, default=1)
    # parser.add_argument("-usb", "--use_beta", type=bool, default=True)
    parser.add_argument("-addTGP", "--addTGP", type=int, default=1)
    parser.add_argument("-SAEbeta", "--SAEbeta", type=float, default=1.0)
    parser.add_argument("-drawtsne", "--drawtsne", type=int, default=1)
    parser.add_argument("-drawround", "--drawround", type=int, default=10)

    # FedGen
    parser.add_argument("-nd", "--noise_dim", type=int, default=512)
    parser.add_argument("-glr", "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=512)
    parser.add_argument("-se", "--server_epochs", type=int, default=100)
    # FML
    parser.add_argument("-al", "--alpha", type=float, default=1.0)
    parser.add_argument("-bt", "--beta", type=float, default=1.0)
    # FedKD
    parser.add_argument("-mlr", "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument("-Ts", "--T_start", type=float, default=0.95)
    parser.add_argument("-Te", "--T_end", type=float, default=0.98)
    # FedGH
    parser.add_argument("-slr", "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument("-mart", "--margin_threthold", type=float, default=100.0)
    # FedKTL
    parser.add_argument(
        "-GPath",
        "--generator_path",
        type=str,
        default="stylegan/stylegan-xl-models/imagenet64.pkl",
    )
    parser.add_argument(
        "-prompt", "--stable_diffusion_prompt", type=str, default="a cat"
    )
    parser.add_argument("-sbs", "--server_batch_size", type=int, default=100)
    parser.add_argument(
        "-gbs",
        "--gen_batch_size",
        type=int,
        default=4,
        help="Not related to the performance. A small value saves GPU memory.",
    )
    parser.add_argument("-mu", "--mu", type=float, default=50.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    if args.use_decay_scheduler:
        print("use scheduler lr decay")

    print("=" * 50)
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))
    print("=" * 50)

    set_seed(42)
    run(args)
