import math
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import time
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel


class AERGCN(nn.Module):
    """Class for the AERGCN model.
    """

    def __init__(self, opt: Any):
        """Instantiate the AERGCN model with the parsed arguments.

        Args:
            opt (Any): Parsed command-line arguments.
        """
        super(AERGCN, self).__init__()
        self.opt = opt

        # text_embeddings1 = AutoModel.from_pretrained(opt.embed_model_name1)

        # if not opt.multi_lingual:
        #     text_embeddings2 = AutoModel.from_pretrained(opt.embed_model_name2)
        # else:
        #     text_embeddings2 = text_embeddings1

        # self.text_embeddings = (text_embeddings1, text_embeddings2)

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        if opt.include_pos_tags:
            self.pos_tag_embeddings1 = nn.Embedding(
                opt.num_pos_tag1, opt.embed_dim, padding_idx=0)
            if opt.lang_1 != opt.lang_2:
                self.pos_tag_embeddings2 = nn.Embedding(
                    opt.num_pos_tag2, opt.embed_dim, padding_idx=0)
            else:
                self.pos_tag_embeddings2 = self.pos_tag_embeddings1

            self.pos_tag_embeddings = (self.pos_tag_embeddings1, self.pos_tag_embeddings2)
        else:
            self.pos_tag_embeddings = (None, None)

        if opt.embed_dim != opt.hidden_dim:
            self.lin_sem1 = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)

            if not opt.multi_lingual:
                self.lin_sem2 = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)
            else:
                self.lin_sem2 = self.lin_sem1

            self.lin_sem = (self.lin_sem1, self.lin_sem2)

            if opt.include_pos_tags:
                self.lin_syn1 = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)

                if opt.lang_1 != opt.lang_2:
                    self.lin_syn2 = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)
                else:
                    self.lin_syn2 = self.lin_syn1

                self.lin_syn = (self.lin_syn1, self.lin_syn2)
            else:
                self.lin_syn = (None, None)

        self.attn_sem1 = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head,
                                   score_function=opt.score_function, dropout=opt.dropout)

        if not opt.multi_lingual:
            self.attn_sem2 = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head,
                                       score_function=opt.score_function, dropout=opt.dropout)
        else:
            self.attn_sem2 = self.attn_sem1

        self.attn_sem = (self.attn_sem1, self.attn_sem2)

        self.attn_syn1 = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head,
                                   score_function=opt.score_function, dropout=opt.dropout)

        if opt.lang_1 != opt.lang_2:
            self.attn_syn2 = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head,
                                       score_function=opt.score_function, dropout=opt.dropout)
        else:
            self.attn_syn2 = self.attn_syn1

        self.attn_syn = (self.attn_syn1, self.attn_syn2)

        self.rgcns1 = R_GCN_module(
            num_layer=opt.num_rgcn_layer,
            in_features=opt.hidden_dim,
            out_features=opt.hidden_dim,
            num_dependencies=opt.num_dependencies1,
            regularization=opt.regularization,
            num_basis=opt.num_basis,
        )

        if opt.lang_1 != opt.lang_2:
            self.rgcns2 = R_GCN_module(
                num_layer=opt.num_rgcn_layer,
                in_features=opt.hidden_dim,
                out_features=opt.hidden_dim,
                num_dependencies=opt.num_dependencies2,
                regularization=opt.regularization,
                num_basis=opt.num_basis,
            )
        else:
            self.rgcns2 = self.rgcns1

        self.rgcns = (self.rgcns1, self.rgcns2)

        self.dense = nn.Linear(opt.hidden_dim * 4, 2)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:
        """Model pipeline for AERGCN.

        Args:
            inputs (dict): {
                'id' (list of str): Sentence pair ids.
                'lemma' (list of str): Target lemmas.
                'pos' (list of str): POS tags of the lemmas.
                'sentence1' (list of str): First sentences.
                'sentence2' (list of str): Second sentences.
                'start1' (list of int): Start indices of the lemmas in the first sentence.
                'end1' (list of int): End indices of the lemmas in the first sentence.
                'start2' (list of int): Start indices of the lemmas in the second sentence.
                'end2' (list of int): End indices of the lemmas in the second sentence.
                'label' (list of str): Labels of the sentence pairs.
                'token1' (list of str): Exact tokens of the lemmas in first sentences.
                'token2' (list of str): Exact tokens of the lemmas in second sentences.
                'alignment1' (list of list of list of int): Alignment between the tokens in the first sentence and the word-pieces encoded by the BERT models.
                'start_offset1' (torch.LongTensor): Indices indicating the start token position of the first sentences (excluding special tokens). 1D tensor.
                'end_offset1' (torch.LongTensor): Indices indicating the end token position of the first sentences (excluding special tokens). 1D tensor.
                'pos_tags1' (torch.LongTensor): POS tag indices corresponding to the tokens in the first sentences. 2D tensor (batch_size, max_sequence_length of sentences).
                'alignment2' (list of list of list of int): Alignment between the tokens in the second sentence and the word-pieces encoded by the BERT models.
                'start_offset2' (torch.LongTensor): Indices indicating the start token position of the second sentences (excluding special tokens). 1D tensor.
                'end_offset2' (torch.LongTensor): Indices indicating the end token position of the second sentences (excluding special tokens). 1D tensor.
                'pos_tags2' (torch.LongTensor): POS tag indices corresponding to the tokens in the second sentences. 2D tensor (batch_size, max_sequence_length of sentences).
                'context_masks1' (torch.LongTensor): Indices used to mask padded tokens for the first sentences. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs).
                'input_ids1' (torch.LongTensor): Token indices for the first sentences. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs).
                'adjacency_tensors1' (tensor.FLoatTensor): Padded encoded adjacency tensors for the first sentences. 4D tensor (batch_size, num_syntactic_relations, max_sequence_length of sentences, max_sequence_length of sentences).
                'context_masks2' (torch.LongTensor): Indices used to mask padded tokens for the second sentences. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs).
                'input_ids2' (torch.LongTensor): Token indices for the second sentences. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs).
                'adjacency_tensors2' (tensor.FLoatTensor): Padded encoded adjacency tensors for the second sentences. 4D tensor (batch_size, num_syntactic_relations, max_sequence_length of sentences, max_sequence_length of sentences).
            }
            fine_tuning (bool, optional): [description]. Defaults to False.

        Returns:
            torch.FloatTensor: [description]
        """
        # tmp_int_time = time.time()
        # for i_lang, lang in enumerate([self.opt.lang_1, self.opt.lang_2], start=1):
        concat_tmp = []
        for i_lang, model_part in enumerate(zip(
                # self.text_embeddings,
                self.pos_tag_embeddings,
                self.lin_sem,
                self.lin_syn,
                self.attn_sem,
                self.attn_syn,
                self.rgcns
        ), start=1):
            # text_embeddings, pos_tag_embeddings, lin_sem, lin_syn, attn_sem, attn_syn, rgcns = model_part
            pos_tag_embeddings, lin_sem, lin_syn, attn_sem, attn_syn, rgcns = model_part
            text_embeddings = self.text_embeddings

            input_ids = inputs[f'input_ids{i_lang}'].to(self.opt.device)

            if fine_tuning:
                text_embeddings.train()
                text = text_embeddings(input_ids)[0]
            else:
                text_embeddings.eval()
                with torch.no_grad():
                    text = text_embeddings(input_ids)[0]
            context_masks = inputs[f'context_masks{i_lang}']

            embedding_len = torch.sum(context_masks, dim=-1)
            if self.opt.embed_dim != self.opt.hidden_dim:
                text_out = lin_sem(text)
            else:
                text_out = text

            sem_attn, _ = attn_sem(text_out, text_out)

            # print('hc: {}s'.format(time.time() - tmp_int_time))
            # tmp_int_time = time.time()

            if not self.opt.include_pos_tags:
                embeddings = text_out
                alignments = inputs[f'alignment{i_lang}']

                aligned_embeddings = list(map(align_embedding, list(embeddings), alignments))
                syntax_text_len = torch.tensor(list(map(len, aligned_embeddings))).to(self.opt.device)
                syntax_text_out = pad_sequence(aligned_embeddings, batch_first=True)

            else:
                syntax_text_len = torch.sum(
                    inputs[f'pos_tags{i_lang}'] != 0, dim=-1).to(self.opt.device)
                syntax_text_out = pos_tag_embeddings(inputs[f'pos_tags{i_lang}'].to(self.opt.device))
                syntax_text_len = torch.sum(
                    inputs[f'pos_tags{i_lang}'] != 0, dim=-1).to(self.opt.device)

                if self.opt.embed_dim != self.opt.hidden_dim:
                    syntax_text_out = lin_syn(syntax_text_out)

            # print('Prepare syntactic branch: {}s'.format(time.time() - tmp_int_time))
            # tmp_int_time = time.time()

            adj = inputs[f'adjacency_tensors{i_lang}'].to(self.opt.device)

            syn_embeddings = rgcns(syntax_text_out, adj)

            syn_attn, _ = attn_syn(syn_embeddings, syn_embeddings)

            # print('hg: {}s'.format(time.time() - tmp_int_time))
            # tmp_int_time = time.time()

            embedding_len = embedding_len.to(device=self.opt.device, dtype=torch.float)

            sem_attn_mean = torch.div(torch.sum(sem_attn, dim=1), embedding_len.view(embedding_len.size(0), 1))
            syn_attn_mean = torch.div(torch.sum(syn_attn, dim=1), syntax_text_len.view(syntax_text_len.size(0), 1))

            concat_tmp.extend([sem_attn_mean, syn_attn_mean])

        concat_embeddings = torch.cat(concat_tmp, dim=-1)
        output = self.dense(concat_embeddings)

        # print('Final classification: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        return output


class AERGCN_no_R(nn.Module):
    """Class for the AERGCN model (replacing RGCN with GCN).
    """

    def __init__(self, opt: Any):
        super(AERGCN_no_R, self).__init__()
        self.opt = opt

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        if opt.include_pos_tags:
            self.pos_tag_embeddings = nn.Embedding(
                opt.num_pos_tag, opt.embed_dim, padding_idx=0)

        if opt.embed_dim != opt.hidden_dim:
            self.lin = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)
        self.attn_k = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head, score_function=opt.score_function,
                                dropout=opt.dropout)  #

        self.attn_q = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head, score_function=opt.score_function,
                                dropout=opt.dropout)  #

        self.gcns = GCN_module(
            num_layer=opt.num_rgcn_layer,
            in_features=opt.hidden_dim,
            out_features=opt.hidden_dim,
        )

        self.attn_k_q = Attention(
            opt.hidden_dim, n_head=opt.head, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_label)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:

        # tmp_int_time = time.time()
        input_ids = inputs['input_ids'].to(self.opt.device)
        if fine_tuning:
            self.text_embeddings.train()
            text = self.text_embeddings(input_ids)[0]
        else:
            self.text_embeddings.eval()
            with torch.no_grad():
                text = self.text_embeddings(input_ids)[0]
        context_masks = inputs['context_masks']

        embedding_len = torch.sum(context_masks, dim=-1)
        if self.opt.embed_dim != self.opt.hidden_dim:
            text_out = self.lin(text)
        else:
            text_out = text

        hid_context = text_out
        hc, _ = self.attn_k(hid_context, hid_context)

        # print('hc: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        if not self.opt.include_pos_tags:
            embeddings = text
            alignments = inputs['alignment']

            aligned_embeddings = list(
                map(align_embedding, list(embeddings), alignments))
            review_text_len = torch.tensor(
                list(map(len, aligned_embeddings))).to(self.opt.device)

            review_text_out = pad_sequence(
                aligned_embeddings, batch_first=True)
        else:
            review_text_len = torch.sum(
                inputs['pos_tags'] != 0, dim=-1).to(self.opt.device)
            review_text_out = self.pos_tag_embeddings(
                inputs['pos_tags'].to(self.opt.device))

        if self.opt.embed_dim != self.opt.hidden_dim:
            review_text_out = self.lin(review_text_out)

        # print('Prepare syntactic branch: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        adj = inputs['adjacency_tensors'].to(self.opt.device)
        adj = torch.sum(adj, dim=1)
        adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))

        x = self.gcns(review_text_out, adj)

        hg, _ = self.attn_q(x, x)

        # print('hg: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        hc_hg, _ = self.attn_k_q(hc, hg)

        embedding_len = embedding_len.to(
            device=self.opt.device, dtype=torch.float)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            embedding_len.view(embedding_len.size(0), 1))

        hc_hg_mean = torch.div(torch.sum(hc_hg, dim=1),
                               review_text_len.view(embedding_len.size(0), 1))

        final_x = torch.cat((hc_hg_mean, hc_mean), dim=-1)

        output = self.dense(final_x)

        # print('Final classification: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        return output


class AERGCN_no_MHA(nn.Module):
    """Class for the AERGCN model (excluding MHA modules).
    """

    def __init__(self, opt: Any):
        super(AERGCN_no_MHA, self).__init__()
        self.opt = opt

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        self.pos_tag_embeddings = nn.Embedding(
            opt.num_pos_tag, opt.embed_dim, padding_idx=0)

        if opt.embed_dim != opt.hidden_dim:
            self.lin = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)

        self.rgcns = R_GCN_module(
            num_layer=opt.num_rgcn_layer,
            in_features=opt.hidden_dim,
            out_features=opt.hidden_dim,
            num_dependencies=opt.num_dependencies,
            regularization=opt.regularization,
            num_basis=opt.num_basis,
        )

        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_label)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:

        # tmp_int_time = time.time()
        input_ids = inputs['input_ids'].to(self.opt.device)
        if fine_tuning:
            self.text_embeddings.train()
            text = self.text_embeddings(input_ids)[0]
        else:
            self.text_embeddings.eval()
            with torch.no_grad():
                text = self.text_embeddings(input_ids)[0]
        context_masks = inputs['context_masks']

        embedding_len = torch.sum(context_masks, dim=-1)
        if self.opt.embed_dim != self.opt.hidden_dim:
            text_out = self.lin(text)
        else:
            text_out = text

        hc = text_out

        # print('hc: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        if not self.opt.include_pos_tags:
            embeddings = text
            alignments = inputs['alignment']

            aligned_embeddings = list(
                map(align_embedding, list(embeddings), alignments))
            review_text_len = torch.tensor(
                list(map(len, aligned_embeddings))).to(self.opt.device)

            review_text_out = pad_sequence(
                aligned_embeddings, batch_first=True)
        else:
            review_text_len = torch.sum(
                inputs['pos_tags'] != 0, dim=-1).to(self.opt.device)
            review_text_out = self.pos_tag_embeddings(
                inputs['pos_tags'].to(self.opt.device))

        if self.opt.embed_dim != self.opt.hidden_dim:
            review_text_out = self.lin(review_text_out)

        # print('Prepare syntactic branch: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        adj = inputs['adjacency_tensors'].to(self.opt.device)

        hg = self.rgcns(review_text_out, adj)

        # print('hg: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        embedding_len = embedding_len.to(
            device=self.opt.device, dtype=torch.float)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            embedding_len.view(embedding_len.size(0), 1))

        hg_mean = torch.div(torch.sum(hg, dim=1),
                            review_text_len.view(embedding_len.size(0), 1))

        final_x = torch.cat((hg_mean, hc_mean), dim=-1)

        output = self.dense(final_x)

        # print('Final classification: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        return output


class AERGCN_no_syn_MHA(nn.Module):
    """Class for the AERGCN model (excluding MHA modules from the syntactic branch).
    """

    def __init__(self, opt: Any):
        super(AERGCN_no_syn_MHA, self).__init__()
        self.opt = opt

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        if opt.include_pos_tags:
            self.pos_tag_embeddings = nn.Embedding(
                opt.num_pos_tag, opt.embed_dim, padding_idx=0)

        if opt.embed_dim != opt.hidden_dim:
            self.lin = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)

        self.attn_k = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head, score_function=opt.score_function,
                                dropout=opt.dropout)  #

        self.rgcns = R_GCN_module(
            num_layer=opt.num_rgcn_layer,
            in_features=opt.hidden_dim,
            out_features=opt.hidden_dim,
            num_dependencies=opt.num_dependencies,
            regularization=opt.regularization,
            num_basis=opt.num_basis,
        )

        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_label)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:

        # tmp_int_time = time.time()
        input_ids = inputs['input_ids'].to(self.opt.device)
        if fine_tuning:
            self.text_embeddings.train()
            text = self.text_embeddings(input_ids)[0]
        else:
            self.text_embeddings.eval()
            with torch.no_grad():
                text = self.text_embeddings(input_ids)[0]
        context_masks = inputs['context_masks']

        embedding_len = torch.sum(context_masks, dim=-1)
        if self.opt.embed_dim != self.opt.hidden_dim:
            text_out = self.lin(text)
        else:
            text_out = text
        hid_context = text_out

        hc, _ = self.attn_k(hid_context, hid_context)

        # print('hc: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        if not self.opt.include_pos_tags:
            embeddings = text
            alignments = inputs['alignment']

            aligned_embeddings = list(
                map(align_embedding, list(embeddings), alignments))
            review_text_len = torch.tensor(
                list(map(len, aligned_embeddings))).to(self.opt.device)

            review_text_out = pad_sequence(
                aligned_embeddings, batch_first=True)
        else:
            review_text_len = torch.sum(
                inputs['pos_tags'] != 0, dim=-1).to(self.opt.device)
            review_text_out = self.pos_tag_embeddings(
                inputs['pos_tags'].to(self.opt.device))

        if self.opt.embed_dim != self.opt.hidden_dim:
            review_text_out = self.lin(review_text_out)

        # print('Prepare syntactic branch: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        adj = inputs['adjacency_tensors'].to(self.opt.device)

        x = self.rgcns(review_text_out, adj)

        # print('hg: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        hc_hg = x

        embedding_len = embedding_len.to(
            device=self.opt.device, dtype=torch.float)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            embedding_len.view(embedding_len.size(0), 1))

        hc_hg_mean = torch.div(torch.sum(hc_hg, dim=1),
                               review_text_len.view(embedding_len.size(0), 1))

        final_x = torch.cat((hc_hg_mean, hc_mean), dim=-1)

        output = self.dense(final_x)

        # print('Final classification: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        return output


class AERGCN_no_MHIA(nn.Module):
    """Class for the AERGCN model (excluding MHIA module).
    """

    def __init__(self, opt: Any):
        super(AERGCN_no_MHIA, self).__init__()
        self.opt = opt

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        if opt.include_pos_tags:
            self.pos_tag_embeddings = nn.Embedding(
                opt.num_pos_tag, opt.embed_dim, padding_idx=0)

        if opt.embed_dim != opt.hidden_dim:
            self.lin = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)

        self.attn_k = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head, score_function=opt.score_function,
                                dropout=opt.dropout)  #

        self.attn_q = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head, score_function=opt.score_function,
                                dropout=opt.dropout)  #

        self.rgcns = R_GCN_module(
            num_layer=opt.num_rgcn_layer,
            in_features=opt.hidden_dim,
            out_features=opt.hidden_dim,
            num_dependencies=opt.num_dependencies,
            regularization=opt.regularization,
            num_basis=opt.num_basis,
        )

        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_label)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:

        # tmp_int_time = time.time()
        input_ids = inputs['input_ids'].to(self.opt.device)
        if fine_tuning:
            self.text_embeddings.train()
            text = self.text_embeddings(input_ids)[0]
        else:
            self.text_embeddings.eval()
            with torch.no_grad():
                text = self.text_embeddings(input_ids)[0]
        context_masks = inputs['context_masks']

        embedding_len = torch.sum(context_masks, dim=-1)
        if self.opt.embed_dim != self.opt.hidden_dim:
            text_out = self.lin(text)
        else:
            text_out = text
        hid_context = text_out

        hc, _ = self.attn_k(hid_context, hid_context)

        # print('hc: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        if not self.opt.include_pos_tags:
            embeddings = text
            alignments = inputs['alignment']

            aligned_embeddings = list(
                map(align_embedding, list(embeddings), alignments))
            review_text_len = torch.tensor(
                list(map(len, aligned_embeddings))).to(self.opt.device)

            review_text_out = pad_sequence(
                aligned_embeddings, batch_first=True)
        else:
            review_text_len = torch.sum(
                inputs['pos_tags'] != 0, dim=-1).to(self.opt.device)
            review_text_out = self.pos_tag_embeddings(
                inputs['pos_tags'].to(self.opt.device))

        if self.opt.embed_dim != self.opt.hidden_dim:
            review_text_out = self.lin(review_text_out)

        # print('Prepare syntactic branch: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        adj = inputs['adjacency_tensors'].to(self.opt.device)

        x = self.rgcns(review_text_out, adj)

        # print('hg: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        hg, _ = self.attn_q(x, x)

        hc_hg = hg

        # print('hg: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        embedding_len = embedding_len.to(
            device=self.opt.device, dtype=torch.float)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            embedding_len.view(embedding_len.size(0), 1))

        hc_hg_mean = torch.div(torch.sum(hc_hg, dim=1),
                               review_text_len.view(embedding_len.size(0), 1))

        final_x = torch.cat((hc_hg_mean, hc_mean), dim=-1)

        output = self.dense(final_x)

        # print('Final classification: {}s'.format(time.time() - tmp_int_time))
        # tmp_int_time = time.time()

        return output


class FullSemantic_no_MHA(nn.Module):
    """Class for the AERGCN model (containing only the semantic branch with plain embeddings).
    """

    def __init__(self, opt: Any):
        super(FullSemantic_no_MHA, self).__init__()
        self.opt = opt

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        if opt.embed_dim != opt.hidden_dim:
            self.lin = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)

        self.dense = nn.Linear(opt.hidden_dim, opt.num_label)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:

        # tmp_int_time = time.time()
        input_ids = inputs['input_ids'].to(self.opt.device)
        if fine_tuning:
            self.text_embeddings.train()
            text = self.text_embeddings(input_ids)[0]
        else:
            self.text_embeddings.eval()
            with torch.no_grad():
                text = self.text_embeddings(input_ids)[0]
        context_masks = inputs['context_masks']

        embedding_len = torch.sum(context_masks, dim=-1)
        if self.opt.embed_dim != self.opt.hidden_dim:
            text_out = self.lin(text)
        else:
            text_out = text

        hc = text_out

        embedding_len = embedding_len.to(
            device=self.opt.device, dtype=torch.float)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            embedding_len.view(embedding_len.size(0), 1))

        output = self.dense(hc_mean)

        return output


class FullSemantic(nn.Module):
    """Class for the AERGCN model (containing only the semantic branch).
    """

    def __init__(self, opt: Any):
        super(FullSemantic, self).__init__()
        self.opt = opt

        self.text_embeddings = AutoModel.from_pretrained(opt.embed_model_name)

        if opt.embed_dim != opt.hidden_dim:
            self.lin = nn.Linear(opt.embed_dim, opt.hidden_dim, bias=True)
        self.attn_k = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=opt.head, score_function=opt.score_function,
                                dropout=opt.dropout)  #

        self.dense = nn.Linear(opt.hidden_dim, opt.num_label)

    def forward(self, inputs: dict, fine_tuning: bool = False) -> torch.FloatTensor:

        # tmp_int_time = time.time()
        input_ids = inputs['input_ids'].to(self.opt.device)
        if fine_tuning:
            self.text_embeddings.train()
            text = self.text_embeddings(input_ids)[0]
        else:
            self.text_embeddings.eval()
            with torch.no_grad():
                text = self.text_embeddings(input_ids)[0]
        context_masks = inputs['context_masks']

        embedding_len = torch.sum(context_masks, dim=-1)
        if self.opt.embed_dim != self.opt.hidden_dim:
            text_out = self.lin(text)
        else:
            text_out = text
        hid_context = text_out

        hc, _ = self.attn_k(hid_context, hid_context)

        embedding_len = embedding_len.to(
            device=self.opt.device, dtype=torch.float)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            embedding_len.view(embedding_len.size(0), 1))

        output = self.dense(hc_mean)

        return output


"""Authored by songyouwei <youwei0314@gmail.com>"""


class Attention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = None, out_dim: int = None, n_head: int = 1, score_function: str = 'dot_product', dropout: float = 0):
        """Class for PyTorch implementation of Multi-head Attention Mechanism.

        Args:
            embed_dim (int): Size of the embeddings.
            hidden_dim (int, optional): Size of the hidden layers. Defaults to None.
            out_dim (int, optional): Size of the outputs. Defaults to None.
            n_head (int, optional): Number of heads. Defaults to 1.
            score_function (str, optional): Score function to use. Defaults to 'dot_product'.
            dropout (float, optional): Dropout probability. Defaults to 0.

        """
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k: torch.FloatTensor, q: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pipeline for the MHA model.

        Args:
            k (torch.FloatTensor): Key vectors.
            q (torch.FloatTensor): Query vectors.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Output embeddings and attention scores.
        """
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous(
        ).view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous(
        ).view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            # (n_head*?, q_len, k_len, hidden_dim*2)
            kq = torch.cat((kxx, qxx), dim=-1)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        # (?, q_len, n_head*hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class GCN_layer(nn.Module):
    """Class for PyTorch implementation of a GCN layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Constructor for the class.

        Args:
            in_features (int): Size of input features.
            out_features (int): Size of output features.
            bias (bool, optional): Flag for bias. Defaults to True.
        """
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text: torch.FloatTensor, adj: torch.FloatTensor) -> torch.FloatTensor:
        """Pipeline for a GCN layer.

        Args:
            text (torch.FloatTensor): Batch of embeddings for the input review sentences. 3D tensor (batch_size, max_sequence_length of reviews, in_features).
            adj (torch.FloatTensor): Batch of adjacency tensors for the input review sentences. 4D tensor (batch_size, num_syntactic_relations, max_sequence_length of reviews, max_sequence_length of reviews).

        Returns:
            torch.FloatTensor: Batch of output embeddings after computation. 3D tensor (batch_size, max_sequence_length of reviews, out_features).
        """
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True)
        intm = torch.matmul(adj, hidden)
        output = intm / \
            torch.where(torch.eq(intm, torch.zeros_like(intm)),
                        torch.ones_like(denom), denom)
        if self.bias is not None:
            return F.relu(output + self.bias, inplace=False)
        else:
            return F.relu(output, inplace=False)


class GCN_module(nn.Module):
    """Class for stacking multiple layers of GCN.
    """

    def __init__(self, num_layer: int, in_features: int, out_features: int, bias: bool = True):
        """Constructor for the class.

        Args:
            num_layer (int): Number of GCN layers to stack.
            in_features (int): Size of input features.
            out_features (int): Size of output features.
            bias (bool, optional): Flag for bias. Defaults to True.
        """
        super(GCN_module, self).__init__()
        self.gcns = nn.ModuleList([GCN_layer(
            in_features=in_features, out_features=out_features, bias=bias) for _ in np.arange(num_layer)])

    def forward(self, x: torch.FloatTensor, adj: torch.FloatTensor) -> torch.FloatTensor:
        """Pipeline for the GCN module.

        Args:
            x (torch.FloatTensor): Batch of embeddings for the input review sentences. 3D tensor (batch_size, max_sequence_length of reviews, in_features).
            adj (torch.FloatTensor): Batch of adjacency tensors for the input review sentences. 4D tensor (batch_size, num_syntactic_relations, max_sequence_length of reviews, max_sequence_length of reviews).

        Returns:
            torch.FloatTensor: Batch of output embeddings after computation. 3D tensor (batch_size, max_sequence_length of reviews, out_features).
        """
        for rgcn in self.gcns:
            x = rgcn(x, adj)
        return x


class R_GCN_layer(nn.Module):
    """Class for the PyTorch implementation of an R-GCN layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_dependencies: int,
        regularization: str = None,
        num_basis: int = 16,
    ):
        """Constructor for the class.

        Args:
            in_features (int): Size of input features.
            out_features (int): Size of output features.
            num_dependencies (int): Number of syntactic relations/dependencies used.
            regularization (str, optional): The weight regularization method to use. Defaults to None.
            num_basis (int, optional): Number of bases for the weight regularization. Defaults to 16.
        """
        super(R_GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regularization = regularization
        if regularization:
            if regularization == 'basis':
                self.coeffs = nn.Parameter(
                    torch.FloatTensor(num_basis, num_dependencies + 1))
                self.basis_weight = nn.Parameter(
                    torch.FloatTensor(in_features, out_features, num_basis))
            elif regularization == 'block':
                q_in = int(self.in_features / num_basis)
                q_out = int(self.out_features / num_basis)
                Qs = torch.FloatTensor(
                    num_basis, num_dependencies + 1, q_out, q_in)
                weight = []
                for dep_ind in np.arange(num_dependencies + 1):
                    Q_dep = torch.block_diag(
                        *torch.unbind(Qs[:, dep_ind, :, :].squeeze()))
                    weight.append(Q_dep)
                self.weight = nn.Parameter(torch.stack(weight, dim=0))
        else:
            # num_dependencies + 1 includes self-loop.
            self.weight = nn.Parameter(
                torch.FloatTensor(num_dependencies + 1, in_features, out_features))
        self.score = nn.Linear(in_features=out_features, out_features=1, bias=True)

    def forward(self, text: torch.FloatTensor, adj: torch.FloatTensor) -> torch.FloatTensor:
        """Pipeline for an R-GCN layer.

        Args:
            text (torch.FloatTensor): Batch of embeddings for the input sentences. 3D tensor (batch_size, max_sequence_length of sentences, in_features).
            adj (torch.FloatTensor): Batch of adjacency tensors for the input sentences. 4D tensor (batch_size, num_syntactic_relations, max_sequence_length of sentences, max_sequence_length of sentences).

        Returns:
            torch.FloatTensor: Batch of output embeddings after computation. 3D tensor (batch_size, max_sequence_length of sentences, out_features).
        """
        text = text.unsqueeze(dim=1)
        if self.regularization:
            if self.regularization == 'basis':
                weight = torch.matmul(
                    self.basis_weight, self.coeffs).permute(2, 0, 1)
            elif self.regularization == 'block':
                weight = self.weight
        else:
            weight = self.weight
        hidden = torch.matmul(text, weight)
        denom = torch.sum(adj, dim=3, keepdim=True)
        intm = torch.matmul(adj, hidden)
        div_output = intm / \
            torch.where(torch.eq(denom, torch.zeros_like(denom)),
                        torch.ones_like(denom), denom)
        r_coeffs = F.softmax(self.score(div_output).squeeze(dim=-1), dim=1)
        output = torch.matmul(div_output.permute(0, 2, 3, 1),
                              r_coeffs.unsqueeze(dim=-1).transpose(1, 2)).squeeze(dim=-1)
        return F.relu(output, inplace=False)


class R_GCN_module(nn.Module):
    """Class for stacking multiple layers of R-GCN.
    """

    def __init__(self, num_layer: int, in_features: int, out_features: int, num_dependencies: int, regularization: str = None, num_basis: int = 16):
        """Constructor for the class.

        Args:
            num_layer (int): Number of R-GCN layers to stack.
            in_features (int): Size of input features.
            out_features (int): Size of output features.
            num_dependencies (int): Number of syntactic relations/dependencies used.
            regularization (str, optional): The weight regularization method to use. Defaults to None.
            num_basis (int, optional): Number of bases for the weight regularization. Defaults to 16.
        """
        super(R_GCN_module, self).__init__()
        self.rgcns = nn.ModuleList(
            [R_GCN_layer(
                in_features=in_features,
                out_features=out_features,
                num_dependencies=num_dependencies,
                regularization=regularization,
                num_basis=num_basis,
            ) for _ in np.arange(num_layer)])

    def forward(self, x: torch.FloatTensor, adj: torch.FloatTensor) -> torch.FloatTensor:
        """Pipeline for the R-GCN module.

        Args:
            x (torch.FloatTensor): Batch of embeddings for the input sentences. 3D tensor (batch_size, max_sequence_length of sentences, in_features).
            adj (torch.FloatTensor): Batch of adjacency tensors for the input sentences. 4D tensor (batch_size, num_syntactic_relations, max_sequence_length of sentences, max_sequence_length of sentences).

        Returns:
            torch.FloatTensor: Batch of output embeddings after computation. 3D tensor (batch_size, max_sequence_length of sentences, out_features).
        """
        for rgcn in self.rgcns:
            x = rgcn(x, adj)
        return x


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Class for PyTorch implementation of the Label Smoothing with Cross Entropy Loss.
    """

    def __init__(self, weight: torch.FloatTensor = None, eps: float = 0.1, reduction: str = 'mean'):
        """Constructor for the class.

        Args:
            weight (torch.FloatTensor, optional): Assigned weights to the classes. Defaults to None.
            eps (float, optional): Smoothing coefficient to defined the probabilty for the incorrect classes. Defaults to 0.1.
            reduction (str, optional): Reduction method, 'None', 'sum' or 'mean'. Defaults to 'mean'.
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.FloatTensor, y: torch.LongTensor) -> torch.FloatTensor:
        """Pipeline for the Label Smoothing Cross Entropy Loss computation.

        Args:
            x (torch.FloatTensor): Classification logits. 2D tensor (batch_size, num_labels).
            y (torch.LongTensor): One-hot encoded labels. 2D tensor (batch_size, num_labels).

        Returns:
            torch.FloatTensor: Loss value.
        """
        n_class = x.shape[-1]
        if self.weight is None:
            self.weight = torch.ones(size=n_class)
        x_hat = self.weight * F.log_softmax(x, dim=-1)
        used_weights = self.weight[y]
        smoothed_loss = -x_hat.sum(dim=-1)
        if self.reduction == 'mean':
            smoothed_loss = smoothed_loss.sum() / used_weights.sum()
        elif self.reduction == 'sum':
            smoothed_loss = smoothed_loss.sum()
        ce_loss = F.nll_loss(x_hat, y, reduction=self.reduction)
        return (1 - self.eps) * ce_loss + self.eps * smoothed_loss / n_class


def align_embedding(embedding: torch.FloatTensor, alignment: list) -> torch.FloatTensor:
    """Align the word-piece embeddings with the token embeddings of the review sentence by summing.

    Args:
        embedding (torch.FloatTensor): Embeddings for the input review sentence. 1D tensor.
        alignment (list of list of int): Alignment between the tokens in the review text and the word-pieces encoded by the BERT models.

    Returns:
        torch.FloatTensor: Summed up embeddings for the review sentences.
    """
    reduced_embedding = []
    for inds in alignment:
        reduced_embedding.append(embedding[inds, :].sum(dim=0))
    return torch.stack(reduced_embedding, dim=0)
