import pickle
import traceback
import spacy
import functools
import numpy as np
import multiprocessing as mp
from collections import Counter
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
from pandas.core.frame import DataFrame
from spacy.language import Language
from scipy.sparse import coo_matrix
from tokenizations import get_alignments
from typing import Tuple
from AERGCN.word_embeddings.embedders import Embedder


def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e
    return wrapper


class DependencyEncoder(object):
    """Class for encoding the syntactic relations/dependencies.
    """

    def __init__(self, resources_dir: Path, lang: str = 'en'):
        """Constructor of the dependency encoder.

        Args:
            nlp (Language): Instantiated model package from SpaCy.
            resources_dir (Path): Path to the resources directory.
            lang (str): Abbreviation for the language. Defaults to 'en' - English.
        """
        self.nlp = select_lang_model(lang_str=lang)
        self.resources_dir = resources_dir
        self.lang = lang
        with open(self.resources_dir.parent / f'{self.lang}' / 'dependencies.txt') as txt_file:
            self.dependencies = txt_file.read().split('\n')
        with open(self.resources_dir.parent / f'{self.lang}' / 'pos_tags.txt') as txt_file:
            self.pos_tags = txt_file.read().split('\n')
        self.dependency2ind = {dep: i for i,
                               dep in enumerate(self.dependencies)}
        with open(self.resources_dir.parent / f'{self.lang}' / 'dependency2ind.dict', 'wb') as b_file:
            pickle.dump(self.dependency2ind, b_file)

    @get_traceback
    def process_single_core(self, proc_id: int, row_inds: np.ndarray, df: DataFrame, embedder: Embedder, text_col: str, tree: bool) -> Tuple[dict, Counter, Counter]:
        num_dependencies = len(self.dependencies)
        adjacency_dict = {}
        dep_list = []
        pos_tag_list = []
        for curr_ind, row_ind in enumerate(row_inds):
            if curr_ind % 100 == 0:
                print(f'Core: {proc_id}, {curr_ind} from {len(row_inds)} rows processed.')
            row = df.iloc[row_ind].to_dict()
            id_ = row['id']
            text = row[text_col]
            # NOTE: Temporary fix for RoBERTa models from huggingface transformers.
            text = text.replace(').', ') .')
            document = self.nlp(text)
            adjacency_dict[id_] = {}
            current_dict = adjacency_dict[id_]
            # NOTE: Could be modified to other columns.
            token_dict = embedder(text)
            input_ids = token_dict['input_ids']
            spacy_tokens = [token.text for token in document]
            transformer_tokens = embedder.tokenizer.convert_ids_to_tokens(
                input_ids[0])
            alignment, _ = get_alignments(spacy_tokens, transformer_tokens)
            current_dict['alignment'] = deepcopy(alignment)
            # Review length after tokenization (-2 for removing special tokens).
            # NOTE: Review length is aligned with the BERT embeddings, which may contain word-piece tokenization that is not consistent with the tokenization of spaCy.
            review_len = len(spacy_tokens)
            embedding_len = input_ids.shape[1] - 2
            # Note down the start and end offsets for the review for later use.
            current_dict['start_offset'] = 1
            current_dict['end_offset'] = embedding_len + 1
            # adjacency_tensor = np.zeros(
            #     shape=(num_dependencies + 1, review_len, review_len), dtype=np.float32)
            adjacency_list = [None for _ in np.arange(num_dependencies + 1)]
            row = [[] for _ in np.arange(num_dependencies)]
            col = [[] for _ in np.arange(num_dependencies)]
            pos_tags = []
            if tree:
                for token in document:
                    dep = token.dep_
                    dep_list.append(dep)
                    dep_ind = self.dependency2ind[dep]
                    # NOTE: Self-loop may be spared for our implementation.
                    # Non-existent syntactic relations do not have adjacency weights.
                    # NOTE 2: Self-loop can be indicated as a separate relation.
                    # if token.i < review_len:
                    #     # adjacency_tensor[dep_ind, :, :] = np.eye(review_len, dtype=np.float32)
                    #     adjacency_tensor[dep_ind, token.head.i, token.i] = 1

                    row[dep_ind].append(token.head.i)
                    col[dep_ind].append(token.i)

                    pos_tag_list.append(token.pos_)
                    pos_tags.append(token.pos_)
                # adjacency_tensor[-1, :, :] = np.eye(review_len, dtype=np.float32)
                # current_dict['adjacency_tensor'] = adjacency_tensor

                for dep_ind in np.arange(num_dependencies):
                    adjacency_list[dep_ind] = coo_matrix(
                        ([1 for _ in np.arange(len(row[dep_ind]))],
                         (row[dep_ind], col[dep_ind])),
                        shape=(review_len, review_len),
                        dtype=np.float32
                    )
                adjacency_list[-1] = coo_matrix(
                    np.eye(review_len, dtype=np.float32))
                current_dict['adjacency_list'] = deepcopy(adjacency_list)
                current_dict['pos_tags'] = deepcopy(pos_tags)
            else:
                for token in document:
                    dep = token.dep_
                    dep_list.append(dep)
                    dep_ind = self.dependency2ind[dep]
                    # NOTE: Self-loop may be spared for our implementation.
                    # Non-existent syntactic relations do not have adjacency weights.
                    # NOTE 2: Self-loop can be indicated as a separate relation.
                    # if token.i < review_len:
                    #     # adjacency_tensor[dep_ind, :, :] = np.eye(review_len, dtype=np.float32)
                    #     adjacency_tensor[dep_ind, token.head.i, token.i] = 1
                    #     adjacency_tensor[dep_ind, token.i, token.head.i] = 1

                    row[dep_ind].append(token.head.i)
                    col[dep_ind].append(token.i)
                    row[dep_ind].append(token.i)
                    col[dep_ind].append(token.head.i)

                    pos_tag_list.append(token.pos_)
                    pos_tags.append(token.pos_)
                # adjacency_tensor[-1, :, :] = np.eye(review_len, dtype=np.float32)
                # current_dict['adjacency_tensor'] = adjacency_tensor

                for dep_ind in np.arange(num_dependencies):
                    adjacency_list[dep_ind] = coo_matrix(
                        ([1 for _ in np.arange(len(row[dep_ind]))],
                         (row[dep_ind], col[dep_ind])),
                        shape=(review_len, review_len),
                        dtype=np.float32
                    )
                adjacency_list[-1] = coo_matrix(
                    np.eye(review_len, dtype=np.float32))
                current_dict['adjacency_list'] = deepcopy(adjacency_list)
                current_dict['pos_tags'] = deepcopy(pos_tags)
        print(f'Core: {proc_id}, all {len(row_inds)} rows processed.')
        return adjacency_dict, Counter(dep_list), Counter(pos_tag_list)

    def __call__(self, df: DataFrame, model_path: str, text_col: str, tree: bool = True) -> Tuple[dict, dict, dict]:
        """Encode the syntactic relations/dependencies for the given dataset using the embedder model specified by model_path.

        Args:
            df (DataFrame): Dataframe of the dataset.
            model_path (str): Name/path of the embedder model.
            text_col (str): Column name of the sentence to parse.
            tree (bool, optional): Flag to indicate whether the syntactic relations/dependencies are encoded as a tree. Defaults to True.

        Returns:
            Tuple[dict, dict, dict]: Dictionary containing the graph information for each sentence;
                Dictionary containing the frequency of each syntactic dependency;
                Dictionary donctaining the frequency of each POS tag.

        """
        embedder = Embedder(model_path)
        dep_counter = Counter()
        pos_tag_counter = Counter()
        adjacency_dict = {}

        cpu_num = mp.cpu_count()
        rows_split = np.array_split(np.arange(df.shape[0]), cpu_num)
        print(f'\n\nProcessing dependency graphs with model: {model_path}')
        print(f'Number of cores: {cpu_num}, dataframe rows per core: {len(rows_split[0])}\n\n')
        # workers = mp.Pool(processes=cpu_num)
        with get_context('spawn').Pool(processes=cpu_num) as workers:
            processes = []
            for proc_id, row_inds in enumerate(rows_split):
                p = workers.apply_async(
                    self.process_single_core, (proc_id, row_inds, df, embedder, text_col, tree))
                processes.append(p)

            workers.close()
            workers.join()

            for p in processes:
                adjacency_dict_single_core, dep_counter_single_core, pos_tag_counter_single_core = p.get()
                dep_counter.update(dep_counter_single_core)
                pos_tag_counter.update(pos_tag_counter_single_core)
                adjacency_dict.update(adjacency_dict_single_core)
                del adjacency_dict_single_core

        return adjacency_dict, dep_counter, pos_tag_counter, embedder.model_name


def prepare_graphs(dataset_resources_dir: Path, df: DataFrame, lang: str, lang_ind: int, embed_model_path: str):
    """Prepare dependency graphs for one entity out of the sentence pairs in the dataset.

    Args:
        dataset_resources_dir (Path): Specified directory of resources for the dataset.
        df (DataFrame): Dataframe of the dataset.
        lang (str): Abbreviation for the language. Defaults to 'en' - English.
        lang_ind (str): Index to indicate the position of the language in the sentence pair.
        embed_model_path (str): BERT model name or path to the BERT model folder.
    """
    dep_enc = DependencyEncoder(resources_dir=dataset_resources_dir.parent, lang=lang)
    adjacency_dict, dep_counter, pos_tag_counter, model_name = dep_enc(
        df=df, model_path=embed_model_path, text_col=f'sentence{lang_ind}')

    with open(dataset_resources_dir / f'{model_name}_{lang_ind}_{lang}.graph', 'wb') as graph_file:
        pickle.dump(adjacency_dict, graph_file)

    with open(dataset_resources_dir / f'dependencies{lang_ind}_{lang}.dict', 'wb') as b_file:
        pickle.dump(dep_counter, b_file)

    with open(dataset_resources_dir / f'pos_tags{lang_ind}_{lang}.dict', 'wb') as b_file:
        pickle.dump(pos_tag_counter, b_file)


def select_lang_model(lang_str: str) -> Language:
    """Select the language model (only supporting spaCy currently).

    Args:
        lang_str (str): abbreviation for language, i.e. 'en' for English

    Returns:
        Language: spaCy language model
    """
    if lang_str == 'en':
        return spacy.load('en_core_web_sm')
    elif lang_str == 'zh':
        return spacy.load('zh_core_web_sm')
    elif lang_str == 'fr':
        return spacy.load('fr_core_news_sm')
    elif lang_str == 'ar':
        raise Exception('Arabic is not supported currently.')
    elif lang_str == 'ru':
        raise Exception('Russian is not supported currently.')
