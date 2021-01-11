# import time
from pathlib import Path
from typing import Tuple

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

    def __call__(self, first_sentence: str, second_sentence: str = None, DEBUG: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pipeline for the tokenizer.

        Args:
            first_sentence (str): First input sentence.
            second_sentence (str): Second input sentence.
            DEBUG (bool, optional): Flag for DEBUG. Defaults to False.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Context mask to indicate the paddings of the sequence (1D torch.FloatTensor) and token ids (1D torch.FloatTensor).
        """
        if second_sentence is not None:
            token_dict = self.tokenizer.encode_plus(
                first_sentence, second_sentence, return_tensors='pt')

        else:
            token_dict = self.tokenizer.encode_plus(
                first_sentence, return_tensors='pt')

        # pad_ind = self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.pad_token)
        # pad_mask = torch.where(token_dict['input_ids'] == pad_ind, torch.ones_like(
        #     token_dict['input_ids']), torch.zeros_like(token_dict['input_ids']))

        context_mask = token_dict['attention_mask']
        input_ids = token_dict['input_ids']

        if not DEBUG:
            return context_mask, input_ids
        else:
            print('\n')
            print(f'model_name: {self.model_name}')
            print(self.tokenizer.__str__)
            print(f"Token (str): {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
            print(f"Token (int): {input_ids}")
            print(f"Context mask: {context_mask}")


class BatchEmbedder(Embedder):
    """Class for wrapping up the batch tokenizer of the BERT models.
    """

    def __call__(self, first_sentences: list, second_sentences: list = None, DEBUG: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pipeline for the batch tokenizer.

        Args:
            first_sentences (list): List of first input sentences.
            second_sentences (list): List of second input sentences.
            DEBUG (bool, optional): Flag for DEBUG. Defaults to False.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Context masks to indicate the paddings of the sequences (2D torch.FloatTensor of (batch_size, max_sequence_length of reviews)) and token ids (2D torch.FloatTensor of (batch_size, max_sequence_length of reviews)).
        """

        # tmp_int_time = time.time()

        if second_sentences is not None:
            sentences_concat = list(zip(first_sentences, second_sentences))
            token_dict = self.tokenizer.batch_encode_plus(
                sentences_concat, return_tensors='pt', padding=True)

        else:
            token_dict = self.tokenizer.batch_encode_plus(
                first_sentences, return_tensors='pt', padding=True)

        context_mask = token_dict['attention_mask']
        input_ids = token_dict['input_ids']

        if not DEBUG:
            return context_mask, input_ids
        else:
            print('\n')
            print(f'model_name: {self.model_name}')
            print(self.tokenizer.__str__)
            print("Token (str): {}".format(
                list(self.tokenizer.convert_ids_to_tokens(input_ids[i]) for i in np.arange(token_dict['input_ids'].shape[0]))))
            print("Token (int): {}".format(list(input_ids[i] for i in np.arange(
                token_dict['input_ids'].shape[0]))))
            print(f"Context mask: {context_mask}")
