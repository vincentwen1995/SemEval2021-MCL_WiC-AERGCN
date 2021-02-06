import pickle
from collections import Counter
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse import spmatrix
from torch.utils.data import Dataset

from AERGCN.dependency_graphs.utils import prepare_graphs
from AERGCN.word_embeddings.embedders import BatchEmbedder


class MCL_WiC_Dataset(Dataset):
    """This class is the Dataset supporting the SemEval 2021 MCL-WiC dataset.
    """

    def __init__(
        self,
        split: str = 'training',
        lang_1: str = 'en',
        lang_2: str = 'en',
        include_pos_tags: bool = False,
        batch_embedder: BatchEmbedder = None,
        k: int = 50,
        dep_filter: list = ['ROOT', 'punct', '', ' '],
        pos_filter: list = ['SYM', 'X', 'SPACE', 'NUM', 'PUNCT'],
    ):
        """Initialize the dataset with the provided arguments.

        Args:
            split (str, optional):  List of choice: 'training', 'development', 'test'. Defaults to 'training'.
            lang_1 (str, optional): Language of the first sentence. Defaults to 'en' - English.
            lang_2 (str, optional): Language of the second sentence. Defaults to 'en' - English.
            batch_embedder (BatchEmbedder, optional): Instantiated batch_embedder to use. Defaults to None.
            force_create (bool, optional): Forceed to create new populated dataframe and save it. Defaults to False.
            k (int, optional): Number of most frequent syntactic relations to include. Defaults to 50.
            #dependency-parsing. Defaults to ['ROOT', 'punct', ''].
            dep_filter (list, optional): Unused syntactic relations/dependencies to filter out. List of syntactic relations can be seen at https://spacy.io/api/annotation
            #pos-tagging. Defaults to ['SYM', 'X', 'SPACE', 'NUM', 'PUNCT'].
            pos_filter (list, optional): Unused POS tags to filter out. List of POS tags can be seen at https://spacy.io/api/annotation
        """
        if batch_embedder is None:
            raise ValueError('An batch_embedder must be provided.')

        self.split = split
        self.lang_1 = lang_1
        self.lang_2 = lang_2
        self.include_pos_tags = include_pos_tags
        self.batch_embedder = batch_embedder
        self.k = k

        dataset_dir = Path(__file__).parent.parent.parent / 'dataset' / f'{self.split}' / 'multilingual' / '{}.{}-{}'.format(
            self.split if self.split == 'training' else self.split[:3], self.lang_1, self.lang_2)
        df_data = pd.read_json(Path(str(dataset_dir) + '.data'))
        df_label = pd.read_json(Path(str(dataset_dir) + '.gold'))
        self.df = pd.merge(df_data, df_label, on='id')

        self.dataset_resources_dir = Path(__file__).parent.parent.parent / \
            'resources' / f'{self.split}' / f'{self.lang_1}-{self.lang_2}'
        self.dataset_resources_dir.mkdir(parents=True, exist_ok=True)

        self.label_ind = {'T': 1, 'F': 0}
        self.inv_label_ind = {1: 'T', 0: 'F'}
        self.classes = {}

        for label, ind in self.label_ind.items():
            self.classes[ind] = self.df[self.df['tag'] == label].shape[0]

        self.df['label'] = self.df['tag'].apply(self.label_ind.get)

        self.graph_dirs = {}
        self.dependencies = {}
        self.dependency2ind = {}
        self.pos_tags2ind = {}
        self.adjacency_dict = {}
        for i_lang, lang in enumerate([self.lang_1, self.lang_2], start=1):

            self.df[f'lemma{i_lang}'] = self.df.apply(
                func=lambda row: row[f'sentence{i_lang}'][row[f'start{i_lang}']: row[f'end{i_lang}']], axis=1)
            # NOTE: Temporary fix for RoBERTa models from huggingface transformers.
            self.df[f'sentence{i_lang}'] = self.df[f'sentence{i_lang}'].apply(
                func=lambda sentence: sentence.replace(").", ") ."))

            graph_dir = self.dataset_resources_dir / f'{self.batch_embedder.model_name}_{i_lang}_{lang}.graph'

            if not graph_dir.exists():
                prepare_graphs(dataset_resources_dir=self.dataset_resources_dir, df=self.df,
                               lang=lang, lang_ind=i_lang, embed_model_path=self.batch_embedder.model_path)

            self.graph_dirs[i_lang] = graph_dir

            filtered_dependencies = Counter()
            with open(self.dataset_resources_dir / f'dependencies{i_lang}_{lang}.dict', 'rb') as b_file:
                dependencies_counter = pickle.load(b_file)
                for dep, count in dependencies_counter.items():
                    if dep not in dep_filter:
                        filtered_dependencies[dep] = count
                most_common_k = filtered_dependencies.most_common(k)
                dependencies, _ = zip(*most_common_k)

            self.dependencies[i_lang] = dependencies

            print(f'Most common {k} dependencies of {self.split}-{i_lang}-{lang}: {most_common_k}')

            with open(self.dataset_resources_dir.parent.parent / f'{lang}' / 'dependency2ind.dict', 'rb') as b_file:
                dependency2ind = pickle.load(b_file)
            self.dependency2ind[i_lang] = dependency2ind

            if include_pos_tags:
                filtered_pos_tags = Counter()
                with open(self.dataset_resources_dir / f'pos_tags{i_lang}_{lang}.dict', 'rb') as b_file:
                    pos_tags_counter = pickle.load(b_file)
                    for pos_tag, count in pos_tags_counter.items():
                        if pos_tag not in pos_filter:
                            filtered_pos_tags[pos_tag] = count
                    pos_tags = filtered_pos_tags.keys()
                    pos_tags2ind = {pos_tag: i for i,
                                    pos_tag in enumerate(pos_tags, start=2)}
                    pos_tags2ind['<PAD>'] = 0
                    pos_tags2ind['<UNK>'] = 1

                self.pos_tags2ind[i_lang] = pos_tags2ind

            with open(graph_dir, 'rb') as graph_file:
                adjacency_dict = pickle.load(graph_file)

            used_dependency_inds = np.array(
                [dependency2ind[dep] for dep in dependencies] + [-1], dtype=np.int)

            aux_dict = {}
            for row_id, row_adjacency_dict in adjacency_dict.items():
                aux_dict[row_id] = deepcopy(row_adjacency_dict)
                adjacency_list_reduced = np.take(
                    row_adjacency_dict['adjacency_list'], used_dependency_inds).tolist()
                aux_dict[row_id]['adjacency_list'] = adjacency_list_reduced

            adjacency_dict.update(aux_dict)
            self.adjacency_dict[i_lang] = adjacency_dict
            del aux_dict

        self.df = self.df.drop(['tag'], axis=1)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i: int) -> dict:
        """Map-style dataset.

        Returns:
            dict: dictionary containing columns for the data sample from the dataframe updated with the corresponding dependency graph.
        """
        output_dict = self.df.iloc[i].to_dict()
        for i_lang, lang in enumerate([self.lang_1, self.lang_2], start=1):
            for key, value in self.adjacency_dict[i_lang][output_dict['id']].items():
                output_dict[f'{key}{i_lang}'] = value
        return output_dict

    def pad(self, sequence: list, max_length: int, pad_ind: int = 0) -> list:
        return sequence + [pad_ind] * (max_length - len(sequence))

    def collate_fn(self, batch: list) -> dict:
        """Stack up the values from the dictionaries of the elements in the batch into lists and contain it in a dictionary.

        Args:
            batch (list):

        Returns:
            dict: dictionary containing stacked up values from the elements in the batch.
        """
        # tmp_int_time = time.time()

        keys = batch[0].keys()
        batch_outputs_dict = {k: [] for k in keys}
        for b in batch:
            for key in keys:
                batch_outputs_dict[key].append(b[key])

        batch_outputs_dict['label'] = torch.LongTensor(batch_outputs_dict['label'])

        token_dict = self.batch_embedder(batch_outputs_dict['sentence1'], batch_outputs_dict['sentence2'])

        batch_outputs_dict['context_masks'], batch_outputs_dict['input_ids'], batch_outputs_dict[
            'offset_mapping'] = token_dict['attention_mask'], token_dict['input_ids'], token_dict['offset_mapping']

        for i_lang, lang in enumerate([self.lang_1, self.lang_2], start=1):
            # batch_outputs_dict[f'context_masks{i_lang}'], batch_outputs_dict[f'input_ids{i_lang}'] = self.batch_embedder(
            #     batch_outputs_dict[f'sentence{i_lang}'], batch_outputs_dict[f'lemma{i_lang}'])
            # Create tensors for the batch output dictionaries.
            batch_outputs_dict[f'start_offset{i_lang}'] = torch.LongTensor(
                batch_outputs_dict[f'start_offset{i_lang}'])
            batch_outputs_dict[f'end_offset{i_lang}'] = torch.LongTensor(
                batch_outputs_dict[f'end_offset{i_lang}'])

            # If using pos tag embeddings, create tensors for the pos tag ids;
            # otherwise, create tensors for the aligned the embeddings (aligning the embeddings from word-piece tokenization with spacy tokenization - sum of word pieces).
            sentence_length = list(map(len, batch_outputs_dict[f'pos_tags{i_lang}']))
            max_length = max(sentence_length)
            if self.include_pos_tags:
                pos_tag_seqs = []
                for seq in batch_outputs_dict[f'pos_tags{i_lang}']:
                    # pos_tag_seqs.append(torch.LongTensor(self.pad(list(map(pos_tags2ind_get, seq)), max_length)))
                    # TODO: Currently using self defined padding method, try making it consistent by using pad_sequence().
                    pos_tag_seqs.append(torch.LongTensor(
                        self.pad(list(map(self.pos_tags2ind[i_lang].get, seq, repeat(1))), max_length)))
                batch_outputs_dict[f'pos_tags{i_lang}'] = torch.stack(pos_tag_seqs, dim=0)

            # print('Prepare pos tags: {}s'.format(time.time() - tmp_int_time))
            # tmp_int_time = time.time()

            adjacency_tensors = []
            adjacency_lists = batch_outputs_dict[f'adjacency_list{i_lang}']
            for i, adjacency_list in enumerate(adjacency_lists):
                adjacency_tensors.append(
                    torch.tensor(
                        np.pad(
                            np.stack(
                                list(map(spmatrix.toarray, adjacency_list)), axis=0),
                            (
                                (0, 0),
                                (0, max_length - sentence_length[i]),
                                (0, max_length - sentence_length[i]),
                            ),
                            'constant'),
                        dtype=torch.float),
                )
            batch_outputs_dict[f'adjacency_tensors{i_lang}'] = torch.stack(
                adjacency_tensors, dim=0)

            # print('Prepare adjacency tensors: {}s'.format(time.time() - tmp_int_time))
            # tmp_int_time = time.time()

            del batch_outputs_dict[f'adjacency_list{i_lang}']

        return batch_outputs_dict
