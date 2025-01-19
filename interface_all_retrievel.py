import argparse
import os

import pandas as pd

from models.Hamming.hamming_retrieve import HammingRetrieve
from models.Hamming_acc.hamming_acc_retrieve import HammingAccRetrieve
from models.all.all_retrieve import AllRetrieve
from ir_measures import *

from models.all_cls.all_cls_retrieve import AllClsRetrieve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--model_name", type=str, default=r"./checkpoints/colbertv2.0")
    parser.add_argument("--root_dir", type=str, default=r"./data")
    parser.add_argument("--save_dir", type=str, default=r"/home/chunming/data/chunming/projects/LSHMultivector")
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
    parser.add_argument("--hamming_threshold", type=int, default=2)
    parser.add_argument("--hash_dimmension", type=int, default=6)
    parser.add_argument("--version", type=str, default="v1")

    args = parser.parse_args()


    # retrieve = AllRetrieve(args)
    # retrieve.setup()
    # # retrieve.show_doc_embedding(0)
    # path =retrieve.retrieve()
    # eval_results = retrieve.evaluation(path)
    # print(eval_results)

    retrieve = AllClsRetrieve(args)
    retrieve.setup()
    # retrieve.show_doc_embedding(0)
    path = retrieve.retrieve()
    eval_results = retrieve.evaluation(path)
    print(eval_results)



    # for hash_dimmension in [5,6,7,8,9,10,11,12,13,14,15,16]:
    #     args.hash_dimmension = hash_dimmension
    #     print(args.hash_dimmension)
    #     retrieve = HammingAccRetrieve(args)
    #     retrieve.run()
    #     eval_results = retrieve.eval_results
    #     eval_results["hash_dimmension"] = str(hash_dimmension)
    #     eval_list.append(eval_results)




