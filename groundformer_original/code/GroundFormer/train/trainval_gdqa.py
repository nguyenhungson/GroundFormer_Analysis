import collections
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import AverageMeter, get_mask


def hierarchical_contrastive_loss(f_v, f_q, f_a, qsns_id, answer_id, temperature=0.07):
    """Hierarchical V-Q-A contrastive learning.

    f_v: [B, D] video embedding
    f_q: [B, Q_n, D] question embeddings (positive at qsns_id)
    f_a: [B, A_n, D] answer embeddings (positive at answer_id)
    """
    B = f_v.size(0)
    _, Q_n, D = f_q.shape
    _, A_n, _ = f_a.shape
    device = f_v.device

    # Level 1: V-Q alignment - (f_v^i, f_q^i) positive, (f_v^i, f_q^j) negative for j != i
    f_v_norm = F.normalize(f_v, p=2, dim=-1)  # [B, D]
    f_q_all = F.normalize(f_q.reshape(B * Q_n, D), p=2, dim=-1)  # [B*Q_n, D]
    sim_vq = torch.matmul(f_v_norm, f_q_all.T) / temperature  # [B, B*Q_n]
    pos_idx_vq = torch.arange(B, device=device) * Q_n + qsns_id  # [B]
    log_prob_vq = F.log_softmax(sim_vq, dim=-1)
    loss_vq = -log_prob_vq[torch.arange(B, device=device), pos_idx_vq].mean()

    # Level 2: Q-A alignment - (f_q^i, f_a^+^i) positive, (f_q^i, f_a^-_k^i) negative
    f_q_pos = f_q[torch.arange(B, device=device), qsns_id, :]  # [B, D]
    f_q_pos_norm = F.normalize(f_q_pos, p=2, dim=-1)
    f_a_norm = F.normalize(f_a.reshape(B * A_n, D), p=2, dim=-1)  # [B*A_n, D]
    sim_qa = torch.matmul(f_q_pos_norm, f_a_norm.T) / temperature  # [B, B*A_n]
    pos_idx_qa = torch.arange(B, device=device) * A_n + answer_id  # [B]
    log_prob_qa = F.log_softmax(sim_qa, dim=-1)
    loss_qa = -log_prob_qa[torch.arange(B, device=device), pos_idx_qa].mean()

    return loss_vq + loss_qa


def eval(model, data_loader, a2v, args, test=False):
    """Answer accuracy + per-frame grounding scores of the predicted answer."""
    model.eval()
    count = 0
    metrics = collections.defaultdict(int)

    results = {}
    ground_res = {}
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            answer_id, qa_tokens, video_frames, qsn_token, question_id = (
                batch["ans_id"],
                batch["qa_tokens"].cuda(),
                batch["video_frames"].cuda(),
                batch["qsn_token"].cuda(),
                batch['question_id'],
            )
            qsns_id = batch['qsns_id']
            video_len = batch["video_len"]

            if args.lan in ['DistilBERT', 'BERT', 'DeBERTa']:
                pad_id = 0
            elif args.lan == 'RoBERTa':
                pad_id = 1

            question_mask = (qsn_token != pad_id).float()
            qa_mask = (qa_tokens != pad_id).float()

            video_mask = get_mask(video_len, video_frames.size(1)).cuda()
            count += answer_id.size(0)

            frame_feats = model.module.video(video_frames, video_mask)
            # at inference the answer pass is not conditioned on the first pass
            model.module.question(qsn_token, frame_feats, question_mask, qsns_id)
            a_logit, amil_score, avideo_embeds, atext_embeds, _, kargs = model.module.answer(
                qa_tokens, None, frame_feats, None, qa_mask, answer_id)

            predicts = a_logit.squeeze()
            predicted = torch.max(predicts, dim=1).indices.cpu()
            metrics["acc"] += (predicted == answer_id).sum().item()

            att = kargs['fatt'].squeeze().cpu().numpy()

            for bs, qid in enumerate(question_id):
                results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer': int(answer_id.numpy()[bs])}
                ground_res[qid] = att[bs].tolist()

    step = "val" if not test else "test"
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results, ground_res


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_acc, running_a_loss = AverageMeter(), AverageMeter()

    for i, batch in enumerate(train_loader):
        answer_id, qa_tokens, video_frames, qsn_token, ans_tokens = (
            batch["ans_id"].cuda(),            # [B] index of the correct answer
            batch["qa_tokens"].cuda(),         # [B, A_n, L] question + answer candidates
            batch["video_frames"].cuda(),      # [B, T, D] frame features
            batch["qsn_token"].cuda(),         # [B, L] the real question
            batch["ans_tokens"].cuda(),
        )

        qsns_id, qsns_tokens = (
            batch['qsns_id'].cuda(),           # [B] index of the real question among the candidates
            batch['qsns_tokens'].cuda(),       # [B, Q_n, L] question candidates
        )
        video_len = batch["video_len"]
        pad_id = tokenizer.pad_token_id

        question_mask = (qsns_tokens != pad_id).float().cuda()
        qa_mask = (qa_tokens != pad_id).float().cuda()

        video_mask = (get_mask(video_len, args.max_feats).cuda() if args.max_feats > 0 else None)

        N = answer_id.size(0)
        label_key = 'qtype_label' if args.qtype_label == 'wh' else 'qtype_label2'
        type_label = torch.stack(batch[label_key]).to('cuda').T.reshape(-1)

        frame_feats = model.module.video(video_frames, video_mask)

        # first pass: question and video
        qtype_logit, t_token, q_logit, qmil_score, qv_attns, video_embeds, qtext_embeds, lang_q, qcma_out = \
            model.module.question(qsns_tokens, frame_feats, question_mask, qsns_id)
        B, _, T = qv_attns.shape
        qv_attn = torch.gather(qv_attns, dim=1, index=qsns_id.unsqueeze(1).unsqueeze(2).expand(B, -1, T))

        # second pass: question + answer candidate, conditioned on the first pass
        a_logit, amil_score, avideo_embeds, atext_embeds, lang_a, av_out = \
            model.module.answer(qa_tokens, t_token, frame_feats, qv_attn, qa_mask, answer_id)

        # QA losses
        answer_loss = criterion(a_logit.squeeze(-1), answer_id)
        question_loss = criterion(q_logit.squeeze(-1), qsns_id)
        type_loss = criterion(qtype_logit.reshape(-1, args.n_qtypes), type_label)

        # hierarchical contrastive loss
        video_embeds = video_embeds.unsqueeze(0) if video_embeds.dim() == 2 else video_embeds
        avideo_embeds = avideo_embeds.unsqueeze(0) if avideo_embeds.dim() == 2 else avideo_embeds
        contra_loss = hierarchical_contrastive_loss(
            video_embeds.mean(dim=1), qtext_embeds.mean(dim=2), atext_embeds.mean(dim=2),
            qsns_id, answer_id, temperature=0.07)

        # MIL loss on both passes
        amil_score = amil_score if amil_score.dim() == 2 else amil_score.unsqueeze(0)
        qmil_score = qmil_score if qmil_score.dim() == 2 else qmil_score.unsqueeze(0)
        mil_loss = criterion(amil_score, answer_id)
        mil_loss += criterion(qmil_score, qsns_id)

        loss = answer_loss + question_loss + type_loss + mil_loss + contra_loss

        predicted = torch.max(a_logit.squeeze(-1), dim=1).indices.cpu()
        running_acc.update((predicted == answer_id.cpu()).sum().item() / N, N)

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_a_loss.update(answer_loss.detach().cpu().item(), N)

        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train acc: {running_acc.avg:.2%}, "
                         f"QA loss: {running_a_loss.avg:.4f}")
            running_acc.reset()
            running_a_loss.reset()
