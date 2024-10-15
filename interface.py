import argparse

from models.all.all_index import AllIndex


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
    parser.add_argument("--index_batch_size", type=int, default=32)
    parser.add_argument("--doc_token", type=str, default="[D]")
    parser.add_argument("--doc_token_id", type=str, default="[unused1]")



    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--query_token", type=str, default="[Q]")
    parser.add_argument("--query_token_id", type=str, default="[unused0]")

    parser.add_argument("--num_doc", type=int, default=3633)

    args = parser.parse_args()

    index = AllIndex(args)
    index.setup()
    index.fit()

