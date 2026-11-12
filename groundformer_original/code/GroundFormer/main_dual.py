import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import logging
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

import eval_ground
from args import get_args
from dataloader.vqa_loader import get_videoqa_loaders
from model.TempVLM import VQA
from tools import getspan
from train.trainval_gdqa import train, eval
from util import compute_a2v, save_to


def _main(local_rank, args):
    rank = local_rank

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', local_rank))
    else:
        world_size = args.world_size if hasattr(args, 'world_size') else torch.cuda.device_count()
        rank = local_rank

    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://' if 'MASTER_ADDR' in os.environ else None,
            rank=rank,
            world_size=world_size
        )

    torch.cuda.set_device(local_rank)
    args.batch_size = args.batch_size // world_size

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if rank == 0:
        if not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
        )
        logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "a")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        logging.info(args)

    if args.lan == 'BERT':
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.lan == 'RoBERTa':
        from transformers import RobertaTokenizerFast
        bert_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif args.lan == 'DeBERTa':
        from transformers import DebertaV2Tokenizer
        bert_tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli')

    a2id, id2a, a2v = None, None, None
    if not args.mc:
        a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=bert_tokenizer,
            amax_words=args.amax_words,
        )
        logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

    model = VQA(
        bert_tokenizer=bert_tokenizer,
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        vocab_size=bert_tokenizer.vocab_size,
        baseline=args.baseline,
        feat_type=args.feat_type,
        lan=args.lan,
        prop_num=args.prop_num,
        sigma=args.sigma,
        n_qtypes=args.n_qtypes,
        token_router=args.token_router,
    ).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        broadcast_buffers=True
    )

    if args.pretrain_path != "":
        checkpoint = torch.load(args.pretrain_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        missing, unexpected = model.module.load_state_dict(checkpoint, strict=False)
        logging.info(f"Loaded checkpoint {args.pretrain_path}")
        if missing:
            logging.warning(f"{len(missing)} missing keys, e.g. {list(missing)[:5]}")
        if unexpected:
            logging.warning(f"{len(unexpected)} unexpected keys, e.g. {list(unexpected)[:5]}")
    if rank == 0:
        logging.info(
            f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_videoqa_loaders(args, args.features_path, a2id, bert_tokenizer, test_mode=args.test, rank=rank, world_size=world_size)

    if args.test != 'train':
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))
    else:
        logging.info("number of train instances: {}".format(len(train_loader.dataset)))
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(local_rank)

    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(
        params_for_optimization, lr=args.lr, weight_decay=args.weight_decay
    )

    if args.test == 'train':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0, len(train_loader) * args.epochs
        )
        logging.info(
            f"Set cosine schedule with {len(train_loader) * args.epochs} iterations"
        )
        if args.pretrain_path != "":
            val_acc, results, _ = eval(model, val_loader, a2v, args, test=False)  # zero-shot VideoQA
            save_path = osp.join(args.save_dir, 'val-res0.json')
            save_to(save_path, results)
        best_val_acc = 0 if args.pretrain_path == "" else val_acc
        best_epoch = 0

        for epoch in range(args.epochs):
            _t0 = time.time()
            train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, bert_tokenizer)
            _t_train = time.time() - _t0

            _t0 = time.time()
            val_acc, results, ground = eval(model, val_loader, a2v, args, test=False)
            _t_eval = time.time() - _t0
            if rank == 0:
                logging.info(f"[Time] Epoch {epoch+1}/{args.epochs}  train={_t_train:.1f}s  eval={_t_eval:.1f}s  total={_t_train+_t_eval:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.module.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
                save_path = osp.join(args.save_dir, 'val-res.json')
                save_to(save_path, results)
                if rank == 0:
                    att_path = osp.join(args.save_dir, 'val_ground_att.json')
                    save_to(att_path, ground)
                    anno_dir = args.load_path
                    getspan.main(args.save_dir, 'val', anno_dir=anno_dir)
                    eval_ground.main(args.save_dir, 'val_ground_ada.json', data_dir=anno_dir, dataset=args.dataset)

            if args.save_ep:
                ep_file = os.path.join(args.save_dir, f"e{epoch}.pth")
                torch.save(model.module.state_dict(), ep_file)
                logging.info('Save to ' + ep_file)

            latest_file = os.path.join(args.save_dir, "latest_model.pth")
            torch.save(model.module.state_dict(), latest_file)
        logging.info(f"Best val model at epoch {best_epoch + 1}")

        # evaluate the best checkpoint (STAR: on val, its test answers are not public)
        args.test = args.eval_split
        args.pretrain_path = os.path.join(args.save_dir, "best_model.pth")
        logging.info(f"Overriding pretrain_path for evaluation: {args.pretrain_path}")
        checkpoint = torch.load(args.pretrain_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        model.module.load_state_dict(checkpoint, strict=False)
        (train_loader, val_loader, test_loader,) = get_videoqa_loaders(
            args, args.features_path, a2id, bert_tokenizer, test_mode=args.test, rank=rank, world_size=world_size)

    test_acc, results, ground = eval(model, test_loader, a2v, args, test=True)
    save_path = osp.join(args.save_dir, f'{args.test}-res.json')
    save_to(save_path, results)

    att_path = osp.join(args.save_dir, f'{args.test}_ground_att.json')
    save_to(att_path, ground)
    anno_dir = args.load_path
    getspan.main(args.save_dir, args.test, anno_dir=anno_dir)
    eval_ground.main(args.save_dir, f'{args.test}_ground_ada.json', data_dir=anno_dir, dataset=args.dataset)


if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if "LOCAL_RANK" in os.environ:
        # launched with torchrun
        _main(int(os.environ["LOCAL_RANK"]), args)
    else:
        world_size = torch.cuda.device_count()
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        if 'MASTER_PORT' not in os.environ:
            # pick a free port so concurrent runs do not collide
            import socket
            with socket.socket() as s:
                s.bind(('', 0))
                os.environ['MASTER_PORT'] = str(s.getsockname()[1])
        if world_size > 1:
            print(f"Using {world_size} GPUs for distributed training via mp.spawn.")
            args.world_size = world_size
            torch.multiprocessing.spawn(_main, args=(args,), nprocs=world_size, join=True)
        else:
            os.environ['WORLD_SIZE'] = '1'
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            _main(0, args)
