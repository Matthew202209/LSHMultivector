import argparse
import os

import pandas as pd
from ir_measures import *

from models.lstm_app.lstm_app_retrieve import LSTMAPPRetrieve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--model_name", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--root_dir", type=str, default=r"./data")
    parser.add_argument("--save_dir", type=str, default=r"/home/chunming/data/chunming/projects/LSHMultivector")
    parser.add_argument("--results_save_to", type=str, default=r"./results")

    parser.add_argument("--dataset", type=str, default=r"nfcorpus")
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

    parser.add_argument("--tree_layers", type=int, default=3)
    parser.add_argument("--first_layer_hash_dim", type=int, default=4)
    parser.add_argument("--hash_dim", type=int, default=8)
    args = parser.parse_args()

    save_dir = r"{}/lstm_app/{}".format(args.results_save_to, args.dataset)
    eval_path = r"{}/{}".format(save_dir, "eval_results")



    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    eval_list = []

    retrieve = LSTMAPPRetrieve(args)
    retrieve.setup()
    retrieve.retrieve()
    # eval_results = retrieve.eval_results
    # eval_results["hash_dimmension"] = str(hash_dimmension)
    # eval_list.append(eval_results)
    #
    # eval_df = pd.DataFrame(eval_list)
    # eval_df.to_csv(r"{}/eval.csv".format(eval_path), index=False)


