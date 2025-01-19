import argparse
import os

import pandas as pd
from ir_measures import *

from models.ann_mvdr.ann_mvdr_retrieve import AnnMvdrRetrieve
from models.isax_MTree.isax_MTree_retrieve import IsaxMTreeRetrieve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--model_name", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--root_dir", type=str, default=r"./data")
    parser.add_argument("--save_dir", type=str, default=r"/home/chunming/data/chunming/projects/LSHMultivector/index")
    parser.add_argument("--results_save_to", type=str, default=r"./results")

    parser.add_argument("--dataset", type=str, default=r"scifact")
    parser.add_argument("--device", type=str, default=r"cpu")



    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--doc_maxlen", type=int, default=300)
    parser.add_argument("--index_batch_size", type=int, default=32)
    parser.add_argument("--doc_token", type=str, default="[D]")
    parser.add_argument("--doc_token_id", type=str, default="[unused1]")



    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--query_token", type=str, default="[Q]")
    parser.add_argument("--query_token_id", type=str, default="[unused0]")
    parser.add_argument("--measure", type=list, default=[nDCG @ 10, RR @ 10, Success @ 10])

    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--KNN", type=int, default=100000)

    parser.add_argument("--low_dim", type=int, default=16)
    parser.add_argument("--sample_p", type=float, default=0.3)
    parser.add_argument("--sax_alphabet_cardinality", type=int, default=5)
    parser.add_argument("--pivot_num", type=int, default=6)
    parser.add_argument("--pivot_random_count", type=int, default=30)
    parser.add_argument("--c_appro", type=float, default=2)
    parser.add_argument("--alpha1", type=float, default=0.01)
    parser.add_argument("--m_num", type=int, default=10)
    parser.add_argument("--mleaf", type=int, default=5)
    parser.add_argument("--search_Radius", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--n_bits_fpq", type=int, default=8)
    args = parser.parse_args()

    save_dir = r"{}/ann_mvdr/{}".format(args.results_save_to, args.dataset)
    eval_path = r"{}/{}".format(save_dir, "eval_results")



    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    eval_list = []

    retrieve = IsaxMTreeRetrieve(args)
    retrieve.setup()
    path = retrieve.retrieve()
    retrieve.save_perf()
    evaluation = retrieve.evaluation(path)
    print(evaluation)
    # eval_results = retrieve.eval_results
    # eval_results["hash_dimmension"] = str(hash_dimmension)
    # eval_list.append(eval_results)
    #
    # eval_df = pd.DataFrame(eval_list)
    # eval_df.to_csv(r"{}/eval.csv".format(eval_path), index=False)


