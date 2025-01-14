import argparse
from sre_parse import parse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niid', type=str, default="noniid", help="non-iid distribution")
    parser.add_argument('--balance', type=str, default="imbalance", help="balance data size per client")
    parser.add_argument('--partition', type=str, default="dir", help="partition distribution, dir|pat｜exdir")
    parser.add_argument('--num_users', type=int, default=40, help="number of users")
    parser.add_argument('--alpha', type=float, default=0.1, help="the degree of imbalance in dir partition")

    parser.add_argument('--class_per_client', type=int, default=4, help="number of classes per client")

    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # text dataset arguments
    parser.add_argument('--max_len', type=int, default=200, help="max length of text")
    parser.add_argument('--max_tokens', type=int, default=32000, help="max number of tokens")

    args = parser.parse_args()
    return args
