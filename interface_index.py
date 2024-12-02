import argparse

from models.Hamming.hamming_index import HammingIndex
from models.Hamming_acc.hamming_acc_index import HammingAccIndex
from models.all.all_index import AllIndex

from bitstring import BitArray
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--model_name", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--root_dir", type=str, default=r"./data")
    parser.add_argument("--save_dir", type=str, default=r"/home/chunming/data/chunming/projects/LSHMultivector")
    parser.add_argument("--dataset", type=str, default=r"nfcorpus")
    parser.add_argument("--device", type=str, default=r"cuda:0")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--doc_maxlen", type=int, default=300)
    parser.add_argument("--index_batch_size", type=int, default=128)
    parser.add_argument("--doc_token", type=str, default="[D]")
    parser.add_argument("--doc_token_id", type=str, default="[unused1]")

    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--query_token", type=str, default="[Q]")
    parser.add_argument("--query_token_id", type=str, default="[unused0]")
    parser.add_argument("--hamming_threshold", type=int, default=2)
    parser.add_argument("--hash_dimmension", type=int, default=6)
    parser.add_argument("--version", type=str, default="v1")

    args = parser.parse_args()

    for dataset in ["nfcorpus"]:
        for version in ["v1"]:
            for hash_dimmension in [10]:
                args.dataset = dataset
                args.version = version
                args.hash_dimmension = hash_dimmension
                print(args.dataset)
                print(args.hash_dimmension)
                print(args.version)
                index = HammingAccIndex(args)
                index.setup()
                index.encode()
                index.indexing()
                index.fit()
        # for hash_dimmension in range(5,17):
        #     args.hash_dimmension = hash_dimmension
        #     print(args.hash_dimmension)
        #     index.fit()

