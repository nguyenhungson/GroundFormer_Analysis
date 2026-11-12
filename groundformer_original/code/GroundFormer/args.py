import argparse
import os

from global_parameters import (
    DEFAULT_DATASET_DIR,
    DEFAULT_STAR_DIR,
    TRANSFORMERS_PATH,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="nextgqa",
        choices=["nextgqa", "nextqa", "star"],
    )

    # Model
    parser.add_argument(
        "--baseline",
        type=str,
        default="dual",
        choices=["dual", "posthoc"],
        help="'dual' is the two-pass GroundFormer model reported in the paper",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="number of layers in the multi-modal transformer",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="number of attention heads in the multi-modal transformer",
    )
    parser.add_argument(
        "--embd_dim",
        type=int,
        default=512,
        help="multi-modal transformer and final embedding dimension",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=2048,
        help="multi-modal transformer feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate in the multi-modal transformer",
    )
    parser.add_argument(
        "--qmax_words",
        type=int,
        default=20,
        help="maximum number of words in the question",
    )
    parser.add_argument(
        "--amax_words",
        type=int,
        default=10,
        help="maximum number of words in the question + answer candidate",
    )
    parser.add_argument(
        "--max_feats",
        type=int,
        default=20,
        help="maximum number of video features considered",
    )

    # Paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="folder holding the QA annotations of each dataset",
    )
    parser.add_argument(
        "--star_data_dir",
        type=str,
        default=DEFAULT_STAR_DIR,
        help="STAR root: {train,val,test}.csv, gsub_*.json, frame2time_*.json, video_feature/CLIPL/*.h5",
    )
    parser.add_argument(
        "--pretrain_path", type=str, default="", help="path to pretrained checkpoint"
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default=TRANSFORMERS_PATH,
        help="path to transformer models checkpoints",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./gmodels/", help="path to save dir"
    )

    # Train
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_val", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_ep", action="store_true", help="save a checkpoint every epoch")
    parser.add_argument(
        "--test", type=str, default='test', choices=['train', 'val', 'test'],
        help="'train' trains and then evaluates, 'val'/'test' only evaluate",
    )
    parser.add_argument(
        "--eval_split", type=str, default='', choices=['', 'val', 'test'],
        help="split evaluated after training. Default: 'test' for NExT-GQA/NExT-QA, "
             "'val' for STAR (the STAR test answers are not public).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.00005, help="initial learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--clip",
        type=float,
        default=12,
        help="gradient clipping",
    )

    # Print
    parser.add_argument(
        "--freq_display", type=int, default=1, help="number of train prints per epoch"
    )
    parser.add_argument(
        "--num_thread_reader", type=int, default=16, help="number of workers"
    )

    parser.add_argument(
        "--mc", type=int, default=5, help="number of answer candidates (NExT-GQA: 5, STAR: 4)"
    )
    parser.add_argument(
        "--feat_type", type=str, default='CLIP', choices=['Swin', 'CLIP', 'CLIPL', 'BLIP', 'BLIP2']
    )
    parser.add_argument(
        "--vg_loss", type=float, default=0,
        help=">0 samples question candidates, required by the first (question) pass",
    )
    parser.add_argument(
        "--lan", type=str, default='RoBERTa', choices=['DistilBERT', 'BERT', 'RoBERTa', 'DeBERTa']
    )
    parser.add_argument(
        "--qtype_label", type=str, default='', choices=['', 'dataset', 'wh'],
        help="supervision of the question-type head. 'dataset': the 8 annotated NExT-GQA "
             "types (CW/CH/TC/TP/TN/DC/DL/DO), 'wh': the 10 wh-types derived from the "
             "question text. Default: 'dataset' for NExT-GQA/NExT-QA, 'wh' for STAR "
             "(STAR has no comparable type annotation), matching the reported models.",
    )
    parser.add_argument(
        "--token_router", type=str, default='', choices=['', 'gate', 'none'],
        help="aggregation of the communication tokens. 'gate': type-token conditioned "
             "routing, 'none': uniform average. Default: 'none' for NExT-GQA/NExT-QA, "
             "'gate' for STAR, matching the reported models.",
    )
    parser.add_argument(
        "--prop_num", type=int, default=1, help="number of temporal proposals"
    )
    parser.add_argument(
        "--sigma", type=float, default=9, help="width of the Gaussian temporal mask"
    )

    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.bert_path

    # feature dimension
    args.feature_dim = args.ff_dim  # CLIP-L: 768
    args.word_dim = 768

    if args.qtype_label == '':
        args.qtype_label = 'wh' if args.dataset == 'star' else 'dataset'
    args.n_qtypes = 10 if args.qtype_label == 'wh' else 8
    if args.token_router == '':
        args.token_router = 'gate' if args.dataset == 'star' else 'none'
    if args.eval_split == '':
        args.eval_split = 'val' if args.dataset == 'star' else 'test'

    if args.dataset == 'star':
        # STAR keeps the annotations next to the features, both under star_data_dir
        star_base = os.path.abspath(os.path.expanduser(args.star_data_dir))
        args.load_path = star_base
        args.train_csv_path = os.path.join(star_base, "train.csv")
        args.val_csv_path = os.path.join(star_base, "val.csv")
        args.test_csv_path = os.path.join(star_base, "test.csv")
        args.vocab_path = os.path.join(star_base, "vocab.json")
        args.features_path = os.path.join(star_base, "video_feature")
        if args.mc == 5:  # STAR has 4 answer candidates
            args.mc = 4
    else:
        load_path = os.path.join(args.dataset_dir, args.dataset)
        args.load_path = load_path
        args.features_path = f'../../../data/{args.dataset}/'
        args.train_csv_path = os.path.join(load_path, "train.csv")
        args.val_csv_path = os.path.join(load_path, "val.csv")
        args.test_csv_path = os.path.join(load_path, "test.csv")
        args.vocab_path = os.path.join(load_path, "vocab.json")

    return args
