import argparse

from models.isax_MTree.isax_MTree_index import IsaxMtreeIndex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--model_name", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--root_dir", type=str, default=r"./data")
    parser.add_argument("--save_dir", type=str, default=r"./index")
    parser.add_argument("--dataset", type=str, default=r"fiqa")
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

    parser.add_argument("--low_dim", type=int, default=16)
    parser.add_argument("--sample_p", type=float, default=0.3)
    parser.add_argument("--sax_alphabet_cardinality", type=int, default=6)
    parser.add_argument("--pivot_num", type=int, default=6)
    parser.add_argument("--pivot_random_count", type=int, default=30)
    parser.add_argument("--c_appro", type=float, default=2)
    parser.add_argument("--alpha1", type=float, default=0.01)
    parser.add_argument("--m_num", type=int, default=8)
    parser.add_argument("--mleaf", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--n_bits_fpq", type=int, default=8)
    args = parser.parse_args()

    for dataset in ["scifact"]:
        args.dataset = dataset
        print(args.dataset)
        index = IsaxMtreeIndex(args)
        index.setup()
        index.encode()
        index.indexing()
