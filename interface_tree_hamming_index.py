import argparse

from models.tree_lsh.tree_hamming_index import TreeHammingIndex





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
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--tree_layers", type=int, default=4)
    parser.add_argument("--hash_dim", type=int, default=6)




    args = parser.parse_args()

    for dataset in ["fiqa"]:
        args.dataset = dataset
        print(args.dataset)
        index = TreeHammingIndex(args)
        index.setup()
        index.encode()
        index.indexing()
        index.save_index()


        # index.encode()
        # index.indexing()
        # for hash_dimmension in range(5,17):
        #     args.hash_dimmension = hash_dimmension
        #     print(args.hash_dimmension)
        #     index.fit()

