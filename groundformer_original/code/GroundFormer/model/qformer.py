import torch
import torch.nn as nn


class MILCrossAttention(nn.Module):
    """Cross-attention with the MIL factorisation of the attention logits:

      det_score = softmax over T      (temporal)
      cls_score = softmax over Q_n    (candidate)
      mil_score = sum_t det_score * cls_score
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, Q_n, gauss_weight=None):
        # query: [B*Q_n, no_query, D];  key, value: [B*Q_n, T, D]
        BQ, nq, D = query.shape
        B = BQ // Q_n
        T = key.shape[1]
        H = self.num_heads
        dk = self.head_dim

        Q = self.W_q(query).view(BQ, nq, H, dk).transpose(1, 2)
        K = self.W_k(key).view(BQ, T, H, dk).transpose(1, 2)
        V = self.W_v(value).view(BQ, T, H, dk).transpose(1, 2)

        raw_logit = (Q @ K.transpose(-1, -2)) / (dk ** 0.5)  # [BQ, H, nq, T]
        attn_weight = torch.softmax(raw_logit, dim=-1)  # [BQ, H, nq, T]

        if gauss_weight is not None:
            gauss_weight = gauss_weight.expand(B, Q_n, T).reshape(BQ, -1).unsqueeze(1).unsqueeze(2)
            attn_weight = attn_weight * (gauss_weight + 1e-10)
            attn_weight = attn_weight / attn_weight.sum(dim=-1, keepdim=True)

        cross_attn_out = attn_weight @ V  # [BQ, H, nq, dk]
        cross_attn_out = cross_attn_out.transpose(1, 2).contiguous().view(BQ, nq, D)
        cross_attn_out = self.W_o(cross_attn_out)  # [BQ, nq, D]

        # MIL path
        raw_logit_avg = raw_logit.mean(dim=1)  # [BQ, nq, T]
        raw_logit_sep = raw_logit_avg.view(B, Q_n, nq, T)  # [B, Q_n, nq, T]

        det_score = torch.softmax(raw_logit_sep, dim=-1)  # over T
        cls_score = torch.softmax(raw_logit_sep, dim=1)   # over Q_n

        mil_score = (cls_score * det_score).sum(dim=-1)  # [B, Q_n, nq]
        grounding_attn = det_score  # [B, Q_n, nq, T]

        return cross_attn_out, grounding_attn, mil_score


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.dense2(self.gelu(self.dense1(x)))


class QFormerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 use_type_token=True, dropout=0.0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.use_type_token = use_type_token

        # SA1: Text -> Tokens
        self.text2token = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn1 = FeedForwardNetwork(hidden_size, intermediate_size, dropout)
        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # SA2: Tokens -> Video
        self.token2video = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn2 = FeedForwardNetwork(hidden_size, intermediate_size, dropout)
        self.ffn_norm2 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Cross-attention: communication tokens attend to the refined video embeddings
        self.cma = MILCrossAttention(hidden_size, num_attention_heads)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ffn3 = FeedForwardNetwork(hidden_size, intermediate_size, dropout)
        self.ffn_norm3 = nn.LayerNorm(hidden_size)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)

    def forward(self, tokens, video_embeds, text_embeds, text_mask=None, L_n=5, gauss_weight=None):
        num_tokens = tokens.size(1)
        B, L_l, D = text_embeds.size()
        seq_len = num_tokens + L_l
        device = tokens.device

        # --- 1. SA1: text -> tokens (text does not attend to the tokens) ---
        tokens_text = torch.cat([tokens, text_embeds], dim=1)

        sa1_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        sa1_mask[num_tokens:, :num_tokens] = True
        sa1_mask = sa1_mask.unsqueeze(0).unsqueeze(0).repeat(B, self.num_attention_heads, 1, 1)

        if text_mask is not None:
            padding_mask = (text_mask == 0).unsqueeze(1).unsqueeze(2)
            sa1_mask[:, :, :, num_tokens:] = sa1_mask[:, :, :, num_tokens:] | padding_mask

        sa1_mask = sa1_mask.view(B * self.num_attention_heads, seq_len, seq_len)

        sa1_out, _ = self.text2token(query=tokens_text, key=tokens_text, value=tokens_text, attn_mask=sa1_mask)
        sa1_out = self.norm1(tokens_text + self.dropout1(sa1_out))
        sa1_out = self.ffn_norm1(sa1_out + self.dropout2(self.ffn1(sa1_out)))

        tokens = sa1_out[:, :num_tokens, :]
        text_embeds = sa1_out[:, num_tokens:, :]

        # --- 2. SA2: tokens -> video (tokens do not attend to the video) ---
        tokens_video = torch.cat([tokens, video_embeds], dim=1)
        sa2_len = tokens_video.size(1)

        sa2_mask = torch.zeros((sa2_len, sa2_len), dtype=torch.bool, device=device)
        sa2_mask[:num_tokens, num_tokens:] = True  # tokens cannot attend to the video

        sa2_out, _ = self.token2video(query=tokens_video, key=tokens_video, value=tokens_video, attn_mask=sa2_mask)
        sa2_out = self.norm2(tokens_video + self.dropout3(sa2_out))
        sa2_out = self.ffn_norm2(sa2_out + self.dropout4(self.ffn2(sa2_out)))

        video_embeds = sa2_out[:, num_tokens:, :]
        tokens = sa2_out[:, :num_tokens, :]

        # --- 3. Cross-attention: tokens attend to the refined video embeddings ---
        query_tokens = tokens

        ca_out, attn_score, mil_score = self.cma(query_tokens, video_embeds, video_embeds, L_n, gauss_weight=gauss_weight)
        ca_out = self.norm3(query_tokens + self.dropout5(ca_out))
        ca_out = self.ffn_norm3(ca_out + self.dropout6(self.ffn3(ca_out)))

        output_tokens = ca_out

        return output_tokens, attn_score, video_embeds, text_embeds, ca_out, mil_score


class QFormer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072,
                 num_hidden_layers=12, num_queries=32, use_type_token=True, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.use_type_token = use_type_token

        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_size))

        if self.use_type_token:
            self.type_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.layers = nn.ModuleList([
            QFormerLayer(hidden_size, num_attention_heads, intermediate_size,
                         use_type_token=self.use_type_token, dropout=dropout)
            for i in range(num_hidden_layers)
        ])

    def forward(self, video_embeds, text_embeds, type_token=None, learnable_token=None, text_mask=None, gauss_weight=None):
        B, L_n, L_l, D = text_embeds.size()

        text_embeds = text_embeds.view(B * L_n, L_l, D)
        text_mask = text_mask.view(B * L_n, L_l)

        video_embeds = video_embeds.repeat_interleave(L_n, dim=0)

        if learnable_token is not None:
            query_tokens = learnable_token.expand(-1, L_n, -1, -1).reshape(B * L_n, self.num_queries, D)
        else:
            query_tokens = self.query_tokens.expand(B * L_n, -1, -1)

        if self.use_type_token:
            if type_token is not None:
                type_token = type_token.expand(-1, L_n, -1).reshape(B * L_n, -1, D)
            else:
                type_token = self.type_token.expand(B * L_n, -1, -1)

            learnable_tokens = torch.cat([type_token, query_tokens], dim=1)
            num_tokens = self.num_queries + 1
        else:
            learnable_tokens = query_tokens
            num_tokens = self.num_queries

        attn_scores = []
        ca_outputs = []
        for layer in self.layers:
            learnable_tokens, attn_score, video_embeds, text_embeds, ca_output, mil_score = layer(
                learnable_tokens, video_embeds, text_embeds, text_mask=text_mask, L_n=L_n, gauss_weight=gauss_weight
            )
            attn_scores.append(attn_score)
            ca_outputs.append(ca_output)

        ca_output = ca_outputs[-1]  # last layer, [B*L_n, num_tokens, D]

        if self.use_type_token:
            type_output = learnable_tokens[:, 0, :].view(B, L_n, -1)
            query_output = learnable_tokens[:, 1:num_tokens, :].view(B, L_n, self.num_queries, -1)
            ca_output = ca_output.view(B, L_n, num_tokens, -1)  # [B, L_n, num_queries + 1, D]
        else:
            type_output = None
            query_output = learnable_tokens[:, :num_tokens, :].view(B, L_n, self.num_queries, -1)
            ca_output = ca_output.view(B, L_n, self.num_queries, -1)

        # average over layers, drop the type token
        attn_scores = torch.stack(attn_scores, dim=-1)  # [B, L_n, nq, T, num_layers]
        if self.use_type_token:
            grounding_scores = attn_scores[:, :, 1:, :, :].mean(-1)  # [B, L_n, num_queries, T]
        else:
            grounding_scores = attn_scores.mean(-1)

        video_embeds = video_embeds.view(B, L_n, -1, D)  # [B, L_n, T, D]
        text_embeds = text_embeds.view(B, L_n, -1, D)  # [B, L_n, L_l, D]
        return type_output, query_output, ca_output, video_embeds, text_embeds, grounding_scores, mil_score
