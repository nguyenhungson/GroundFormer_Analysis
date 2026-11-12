import torch.nn as nn


class Bert(nn.Module):
    """Finetuned language backbone."""

    def __init__(self, tokenizer, lan):
        super(Bert, self).__init__()

        if lan == 'DistilBERT':
            from transformers import DistilBertModel, DistilBertConfig
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        elif lan == 'BERT':
            from transformers import BertModel, BertConfig
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
            self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        elif lan == 'RoBERTa':
            from transformers import RobertaModel, RobertaConfig
            config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
            self.bert = RobertaModel.from_pretrained("roberta-base", config=config)
        elif lan == 'DeBERTa':
            from transformers import DebertaV2Config, DebertaV2Model
            config = DebertaV2Config.from_pretrained("microsoft/deberta-v2-xlarge-mnli", output_hidden_states=True)
            self.bert = DebertaV2Model.from_pretrained("microsoft/deberta-v2-xlarge-mnli", config=config)

        self.tokenizer = tokenizer

    def forward(self, tokens):
        attention_mask = (tokens != self.tokenizer.pad_token_id).float()
        outs = self.bert(tokens, attention_mask=attention_mask)
        return outs[0]


class LanModel(nn.Module):
    """Language embedding module."""

    def __init__(self, tokenizer, lan, word_dim=768, out_dim=512):
        super(LanModel, self).__init__()

        self.bert = Bert(tokenizer, lan)
        self.linear_text = nn.Linear(word_dim, out_dim)

    def forward(self, txt, is_answer=False):
        # multiple-choice input: [B, n_choices, n_tokens]
        bs, nans, lans = txt.shape

        txt = txt.view(bs * nans, lans)
        pad_mask = (txt != self.bert.tokenizer.pad_token_id).float().view(bs, nans, lans)

        lang_l = self.bert(txt)  # [B*n_choices, n_tokens, D]
        lang_l = self.linear_text(lang_l)

        lang_l = lang_l.view(bs, nans, lans, -1)
        lang_g = lang_l.mean(2)
        return lang_g, lang_l, pad_mask
