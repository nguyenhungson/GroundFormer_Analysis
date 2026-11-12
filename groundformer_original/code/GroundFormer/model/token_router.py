import torch.nn as nn
import torch.nn.functional as F


class AttentionReweightingRouter(nn.Module):
    """Weights the communication tokens by the type token.

    The weights aggregate both the token embeddings and their temporal attention.
    """

    def __init__(self, hidden_size, num_queries, use_gating=True, temperature=1.0):
        """
        Args:
            hidden_size: embedding dimension
            num_queries: number of communication tokens (experts)
            use_gating: True for the 2-layer gating network, False for a linear projection
            temperature: softmax temperature (lower is sparser)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.use_gating = use_gating
        self.temperature = temperature

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_size, num_queries)
            )
        else:
            self.gate = nn.Linear(hidden_size, num_queries)

    def forward(self, type_token, query_tokens, attn_scores=None):
        """
        Args:
            type_token:  [B, N, D] type token (question / answer type information)
            query_tokens:[B, N, num_queries, D] communication tokens
            attn_scores: [B, N, num_queries, T] per-token temporal attention (optional)

        Returns:
            aggregated_token: [B, N, D] weighted sum of the communication tokens
            query_weights:    [B, N, num_queries] routing weights
            reweighted_attn:  [B, N, T] routed temporal attention
        """
        B, N, D = type_token.size()

        query_logits = self.gate(type_token)  # [B, N, num_queries]
        query_weights = F.softmax(query_logits / self.temperature, dim=-1)

        weighted_queries = query_tokens * query_weights.unsqueeze(-1)  # [B, N, num_queries, D]
        aggregated_token = weighted_queries.sum(dim=2)  # [B, N, D]

        reweighted_attn = None
        if attn_scores is not None:
            if attn_scores.dim() == 4:
                reweighted_attn = (attn_scores * query_weights.unsqueeze(-1)).sum(dim=2)  # [B, N, T]
            elif attn_scores.dim() == 3:
                reweighted_attn = attn_scores  # [B, N, T]
        return aggregated_token, query_weights, reweighted_attn
