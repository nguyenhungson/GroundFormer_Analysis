import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import gelu
from transformers import BertConfig, RobertaConfig, DebertaConfig

from .embeddings import Embeddings
from .language_model import LanModel
from .qformer import QFormer
from .token_router import AttentionReweightingRouter


def gaussian_smooth(attention_scores, sigma=3.0):
    """Temporal smoothing of the grounding scores with a 1D Gaussian kernel.

    attention_scores: [B, Q_n, T]
    """
    B, Q_n, T = attention_scores.shape

    kernel_size = int(3 * sigma) // 2 * 2 + 1  # force an odd kernel size
    x = torch.arange(kernel_size, device=attention_scores.device, dtype=attention_scores.dtype) - kernel_size // 2
    gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    attn_flat = attention_scores.view(B * Q_n, 1, T)
    kernel = gaussian_kernel.view(1, 1, -1)

    smoothed = F.conv1d(attn_flat, kernel, padding=kernel_size // 2)  # [B * Q_n, 1, T]
    smoothed = smoothed.view(B, Q_n, T)
    smoothed = smoothed / (smoothed.max(dim=-1, keepdim=True)[0] + 1e-8)
    return smoothed


class VQA(nn.Module):
    def __init__(
        self,
        bert_tokenizer,
        feature_dim=1024,
        word_dim=768,
        N=2,
        h=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        Q=20,
        T=20,
        vocab_size=50265,
        baseline="",
        n_negs=1,
        feat_type='CLIP',
        lan='RoBERTa',
        prop_num=1,
        sigma=9,
        num_segments=4,
        n_qtypes=8,
        token_router='gate',
    ):
        """
        :param feature_dim: dimension of the input video features
        :param word_dim: dimension of the input question features
        :param N: number of transformer layers
        :param h: number of transformer heads
        :param d_model: dimension for the transformer and final embedding
        :param d_ff: hidden dimension in the transformer
        :param dropout: dropout rate in the transformer
        :param Q: maximum number of tokens in the question
        :param T: maximum number of video features
        :param vocab_size: size of the vocabulary for the masked language modeling head
        """
        super(VQA, self).__init__()
        self.baseline = baseline
        self.Q = Q
        self.T = T
        self.n_negs = n_negs
        self.mc = 5
        self.sigma = sigma
        self.num_segments = num_segments
        d_pos = 128

        self.linear_video = nn.Linear(feature_dim, d_model)
        self.norm_video = nn.LayerNorm(d_model, eps=1e-12)

        self.position_v = Embeddings(d_model, 0, T, dropout, True, d_pos)

        if lan == 'BERT':
            pt_config_name = 'bert-base-uncased'
            config = BertConfig
        elif lan == 'RoBERTa':
            pt_config_name = 'roberta-base'
            config = RobertaConfig
        elif lan == 'DeBERTa':
            pt_config_name = 'microsoft/deberta-v2-xlarge-mnli'
            config = DebertaConfig

        self.config = config.from_pretrained(
            pt_config_name,
            num_hidden_layers=N,
            hidden_size=d_model,
            attention_probs_dropout_prob=dropout,
            intermediate_size=d_ff,
            num_attention_heads=h,
        )

        self.apply(self._init_weights)
        self.answer_embeddings = None

        self.lang_model = LanModel(bert_tokenizer, lan=lan, word_dim=d_model, out_dim=d_model)

        self.no_query = 32
        self.qformer = QFormer(num_hidden_layers=4, num_attention_heads=12, hidden_size=d_model,
                               intermediate_size=d_model * 4, num_queries=self.no_query,
                               use_type_token=True, dropout=0.0)
        self.router = AttentionReweightingRouter(d_model, self.no_query) if token_router == 'gate' else None

        self.question_classifier = nn.Linear(d_model, 1)
        self.answer_classifier = nn.Linear(d_model, 1)
        self.type_classifier = nn.Linear(d_model, n_qtypes)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def video(self, video, video_mask):
        video_feats = self.linear_video(video)
        video_feats = gelu(video_feats)
        video_feats = self.norm_video(video_feats)
        video_feats = self.position_v(video_feats)
        return video_feats

    def question(self, qsns_tokens, frame_feats, question_mask, qsns_id):
        """First pass: question and video."""
        B, T, D = frame_feats.size()
        qsns_tokens = qsns_tokens.unsqueeze(1) if qsns_tokens.dim() != 3 else qsns_tokens
        B, Q_n, Q_l = qsns_tokens.size()

        lang_g, lang_l, padding = self.lang_model(qsns_tokens)
        t_token, l_tokens, cma_out, video_embeds, text_embeds, attn_score, mil_score = self.qformer(
            frame_feats, lang_l, type_token=None, learnable_token=None, text_mask=question_mask)

        # index 0 is the type token, 1: are the communication tokens
        t_token = cma_out[:, :, 0, :]
        cma_outs = cma_out[:, :, 1:, :]
        mil_score = mil_score[:, :, 1:]

        if self.router is not None:
            cma_out, router_score, attn_score = self.router(t_token, cma_outs, attn_score)
            mil_score = (mil_score * router_score).sum(dim=2)
        else:
            cma_out = cma_outs.mean(2)
            attn_score = attn_score.mean(2)
            mil_score = mil_score.mean(2)
        attn_score = gaussian_smooth(attn_score, sigma=2.0)

        question_logits = self.question_classifier(cma_out)
        type_logit = self.type_classifier(t_token)

        cma_outs = cma_outs if not self.training else torch.gather(cma_outs, dim=1, index=qsns_id.view(-1, 1, 1, 1).expand(B, -1, cma_outs.size(2), D))
        t_token = t_token if not self.training else torch.gather(t_token, dim=1, index=qsns_id.view(-1, 1, 1).expand(B, -1, D))
        video_embeds = video_embeds.squeeze() if not self.training else torch.gather(video_embeds, dim=1, index=qsns_id.view(-1, 1, 1, 1).expand(B, -1, T, D)).squeeze()

        return type_logit, t_token, question_logits, mil_score, attn_score, video_embeds, text_embeds, lang_g, cma_outs

    def answer(self, qa_tokens, t_token, frame_feats, qv_attn, answer_mask, answer_id):
        """Second pass: question + answer candidate and video, conditioned on the first pass."""
        B, T, D = frame_feats.size()
        B, A_n, A_l = qa_tokens.size()

        lang_g, lang_l, padding = self.lang_model(qa_tokens)
        t_token, l_tokens, cma_out, video_embeds, text_embeds, attn_score, mil_score = self.qformer(
            frame_feats, lang_l, type_token=t_token, learnable_token=None,
            text_mask=answer_mask, gauss_weight=qv_attn)

        t_token = cma_out[:, :, 0, :]
        cma_outs = cma_out[:, :, 1:, :]
        mil_score = mil_score[:, :, 1:]

        if self.router is not None:
            cma_out, router_score, attn_score = self.router(t_token, cma_outs, attn_score)
            mil_score = (mil_score * router_score).sum(dim=2)
        else:
            cma_out = cma_outs.mean(2)
            attn_score = attn_score.mean(2)
            mil_score = mil_score.mean(2)
        attn_score = gaussian_smooth(attn_score, sigma=3.0)

        answer_logits = self.answer_classifier(cma_out)

        pred = answer_id if self.training else answer_logits.squeeze().argmax(dim=-1)
        attn_score = attn_score if self.training else torch.gather(attn_score, dim=1, index=pred.unsqueeze(1).unsqueeze(2).expand(B, -1, T))
        video_embeds = video_embeds.squeeze() if not self.training else torch.gather(video_embeds, dim=1, index=answer_id.view(-1, 1, 1, 1).expand(B, -1, T, D)).squeeze()

        outs = {'fatt': attn_score}
        return answer_logits, mil_score, video_embeds, text_embeds, lang_g, outs
