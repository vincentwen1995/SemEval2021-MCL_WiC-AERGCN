# import time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer


class Embedder(object):
    """Class for wrapping up the tokenizer of the BERT models.
    """

    def __init__(self, model_path: str = 'xlm-roberta-base'):
        """Constructor of the class.

        Args:
            model_path (str, optional): BERT model name or path to the BERT model folder. Defaults to 'xlm-roberta-base'.
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if any([r'\\' in model_path, '/' in model_path, '\\' in model_path]):
            self.model_name = '-'.join(Path(model_path).stem.split('-')[:-4])
        else:
            self.model_name = model_path

    def __call__(self, first_sentence: str, second_sentence: str = None, DEBUG: bool = False) -> Dict[str, torch.Tensor]:
        """Pipeline for the tokenizer.

        Args:
            first_sentence (str): First input sentence.
            second_sentence (str): Second input sentence.
            DEBUG (bool, optional): Flag for DEBUG. Defaults to False.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Context mask to indicate the paddings of the sequence (1D torch.FloatTensor) and token ids (1D torch.FloatTensor).
            Dict[str, torch.Tensor]: {
                'input_ids' (torch.LongTensor): Token indices for the sentences. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs).
                'attention_mask' (torch.LongTensor): Indices used to mask padded tokens for the sentences. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs). 
                'offset_mappping' (torch.LongTensor): Character indices for each token in the sentences. 3D tensor (batch_size, max_sequence_length of concatenated sentence pairs, 2)
            }
        """
        if second_sentence is not None:
            token_dict = self.tokenizer(first_sentence, second_sentence, return_tensors='pt')

        else:
            token_dict = self.tokenizer(first_sentence, return_tensors='pt')

        # pad_ind = self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.pad_token)
        # pad_mask = torch.where(token_dict['input_ids'] == pad_ind, torch.ones_like(
        #     token_dict['input_ids']), torch.zeros_like(token_dict['input_ids']))

        if not DEBUG:
            return token_dict
        else:
            print('\n')
            print(f'model_name: {self.model_name}')
            print(self.tokenizer.__str__)
            print(f"Token (str): {self.tokenizer.convert_ids_to_tokens(token_dict['input_ids'][0])}")
            print(f"Token (int): {token_dict['input_ids']}")
            print(f"Context mask: {token_dict['attention_mask']}")

    """Authored by Yury Kashnitsky <kashnitsky @ Kaggle>"""

    @staticmethod
    def get_indices_of_the_words_by_start_char_id(offset_mapping: List[Tuple[int, int]],
                                                  start_char_id1: int,
                                                  start_char_id2: int):
        """
        :param offset_mapping: a list of tuples, each of them indicating start and end ids 
                                (in the original string) of word 
                                pieces after tokenization with XLMRobertaTokenizerFast
        :param start_char_id1: start id in the first sentence
        :param start_char_id2: start id in the second sentence

        """

        zero_idx = [i for i, (s, e) in enumerate(offset_mapping) if (s, e) == (0, 0)]

        offset_mapping_first_sent = offset_mapping[:zero_idx[2]]
        offset_mapping_second_sent = offset_mapping[zero_idx[2]:]

        id1, id2 = 0, 0
        for i, (s, e) in enumerate(offset_mapping_first_sent):
            if (s, e) == (0, 0):
                continue
            # Check if the offset mappings start with the char index or include it (this is for cases where the tokenization creates deviation, i.e. training.en-en.7353 sentence2).
            if (s == start_char_id1) or (start_char_id1 > s and start_char_id1 <= e):
                # Find the last token (ordered) that has the start index as specified.
                # This is to avoid some cases where the returned offset mappings are troublesome.
                # For example, when encoding training.en-en.914, two entities of the offset mappings are [7, 8] and [7, 16], where [7, 8] is trivial.
                id1 = i
                # break

        for i, (s, e) in enumerate(offset_mapping_second_sent):
            if (s, e) == (0, 0):
                continue
            if (s == start_char_id2) or (start_char_id2 > s and start_char_id2 <= e):
                id2 = i
                # break

        # The saved alignments are separate for the sentence pairs,
        # so for this application we only need id2 counting from the start without offset.
        # id2 += len(offset_mapping_first_sent)

        return id1, id2


class BatchEmbedder(Embedder):
    """Class for wrapping up the batch tokenizer of the BERT models.
    """

    def __call__(self, first_sentences: list, second_sentences: list = None, DEBUG: bool = False) -> Dict[str, torch.Tensor]:
        """Pipeline for the batch tokenizer.

        Args:
            first_sentences (list): List of first input sentences.
            second_sentences (list): List of second input sentences.
            DEBUG (bool, optional): Flag for DEBUG. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: {
                'input_ids' (torch.LongTensor): Token indices for the sentence pairs. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs).
                'attention_mask' (torch.LongTensor): Indices used to mask padded tokens for the sentence pairs. 2D tensor (batch_size, max_sequence_length of concatenated sentence pairs). 
                'offset_mapping' (torch.LongTensor): Character indices for each token in the sentence pairs. 3D tensor (batch_size, max_sequence_length of concatenated sentence pairs, 2)
            }
        """

        # tmp_int_time = time.time()

        if second_sentences is not None:
            token_dict = self.tokenizer(first_sentences, second_sentences, return_tensors='pt',
                                        return_offsets_mapping=True, padding=True)
        else:
            token_dict = self.tokenizer(
                first_sentences, return_tensors='pt', return_offsets_mapping=True, padding=True)

        if not DEBUG:
            return token_dict
        else:
            print('\n')
            print(f'model_name: {self.model_name}')
            print(self.tokenizer.__str__)
            print("Token (str): {}".format(
                list(self.tokenizer.convert_ids_to_tokens(token_dict['input_ids'][i]) for i in np.arange(token_dict['input_ids'].shape[0]))))
            print("Token (int): {}".format(list(token_dict['input_ids'][i] for i in np.arange(
                token_dict['input_ids'].shape[0]))))
            print(f"Context mask: {token_dict['attention_mask']}")
