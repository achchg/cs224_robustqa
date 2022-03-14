import argparse
from re import X
from xmlrpc.client import boolean

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--ideal_batch_size', type=int, default = 20)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--oodomain', type=boolean, default = True)
    parser.add_argument('--oodomain_train-dir', type=str, default='datasets/oodomain_train')
    parser.add_argument('--indomain_eval', type=boolean, default = False)
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', default = True) #action='store_true'
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--outer_batch-size', type=int, default=16)
    parser.add_argument('--outer_num_epochs', type=int, default=1000)
    parser.add_argument('--outer_lr', type=float, default=1e-5)
    parser.add_argument('--n_task', type=float, default=10)
    parser.add_argument('--p_oodomain_task', type=float, default=1)
    parser.add_argument('--k_shot', type=float, default=20)
    parser.add_argument('--sample_p', type=float, default=1)
    args = parser.parse_args()
    return args
