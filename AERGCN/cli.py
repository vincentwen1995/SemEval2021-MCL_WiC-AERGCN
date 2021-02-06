import datetime
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator
from sklearn.metrics import f1_score, accuracy_score

import numpy as np
import spacy
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from AERGCN.data.datasets import MCL_WiC_Dataset
from AERGCN.input_pipeline.models import LabelSmoothingCrossEntropyLoss
from AERGCN.word_embeddings.embedders import BatchEmbedder

try:
    import wandb
except ImportError:
    print('wandb not installed. Please install wandb for better logging functionality.')
    wandb = None
else:
    if all([hasattr(wandb, '__path__'), getattr(wandb, '__file__', None) is None]):
        print('wandb not installed. Please install wandb for better logging functionality.')
        wandb = None


class Interface:
    """Class for providing the CLI functionality.
    """

    def __init__(self, opt: Any):
        """Constructor of the class.

        Args:
            opt (Any): Parsed command-line arguments.
        """
        self.opt = opt
        self.main_dir = Path(__file__).parent.parent
        if self.opt.log_dir is None:
            self.model_num = 0
            self.log_parent_dir = self.main_dir / 'results' / \
                f'{self.opt.model_name}' / \
                f'{datetime.date.today()}'
            self.model_num += len([path.stem for path in self.log_parent_dir.glob('*')])
            self.log_dir = self.log_parent_dir / f'{self.model_num}'
        else:
            self.log_dir = Path(self.opt.log_dir)

        print('Arguments:')
        self.arg_dict = {}
        for arg in vars(self.opt):
            print(f'>>> {arg}: {getattr(self.opt, arg)}')
            self.arg_dict[str(arg)] = str(getattr(self.opt, arg))
        if wandb:
            wandb.init(name=f'{self.opt.model_name}-{self.opt.mode}-{datetime.date.today()}-{self.model_num}',
                       sync_tensorboard=True, config=self.arg_dict)

        if self.opt.force_new:
            def rm_tree(pth):
                for child in pth.glob('*'):
                    if child.is_file():
                        child.unlink()
                    else:
                        rm_tree(child)
                pth.rmdir()
            if self.log_dir.exists():
                rm_tree(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if opt.fine_tuning:
            # if any([r'\\' in opt.embed_model_name1, '/' in opt.embed_model_name1, '\\' in opt.embed_model_name1]):
            #     embed_model = '-'.join(
            #         Path(opt.embed_model_name1).stem.split('-')[:-4])
            # else:
            #     embed_model = opt.embed_model_name1
            if any([r'\\' in opt.embed_model_name, '/' in opt.embed_model_name, '\\' in opt.embed_model_name]):
                embed_model = '-'.join(
                    Path(opt.embed_model_name).stem.split('-')[:-4])
            else:
                embed_model = opt.embed_model_name
            self.ft_dir = self.log_dir / f'{embed_model}-ft'
            self.ft_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / 'checkpoints'
        self.best_model_dir = self.log_dir / 'best_state_dict'
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        self.batch_embedder = BatchEmbedder(
            model_path=opt.embed_model_name)
        self.nlp = spacy.load('en_core_web_sm')

        # if self.opt.label_smoothing:
        #     Loss = LabelSmoothingCrossEntropyLoss
        # else:
        #     # Loss = nn.CrossEntropyLoss
        Loss = nn.BCEWithLogitsLoss

        self.semeval_train = MCL_WiC_Dataset(
            split='training',
            lang_1=opt.lang_1,
            lang_2=opt.lang_2,
            include_pos_tags=opt.include_pos_tags,
            batch_embedder=self.batch_embedder,
            k=opt.num_dependencies,
        )

        if self.opt.mode == 'training':
            self.train_data_loader = DataLoader(
                dataset=self.semeval_train,
                batch_size=opt.batch_size,
                shuffle=True,
                collate_fn=self.semeval_train.collate_fn,
                pin_memory=True,
            )

            self.semeval_dev = MCL_WiC_Dataset(
                split='development',
                lang_1=opt.lang_1,
                lang_2=opt.lang_2,
                include_pos_tags=opt.include_pos_tags,
                batch_embedder=self.batch_embedder,
                k=opt.num_dependencies,
            )

            self.dev_data_loader = DataLoader(
                dataset=self.semeval_dev,
                # TODO: Check whether it could be increased further (dependent on computaitonal resource).
                batch_size=opt.batch_size * 4,
                shuffle=False,
                collate_fn=self.semeval_dev.collate_fn,
                pin_memory=True,
            )

            if self.opt.include_pos_tags:
                self.opt.num_pos_tag1 = len(self.semeval_train.pos_tags2ind[1])
                self.opt.num_pos_tag2 = len(self.semeval_train.pos_tags2ind[2])

            self.opt.num_label = len(self.semeval_train.inv_label_ind)

            self.train_class_weights = [float(self.semeval_train.classes[i]) / len(self.semeval_train)
                                        for i in np.arange(len(self.semeval_train.classes))]
            self.train_class_weights_repri = torch.FloatTensor(
                self.train_class_weights).reciprocal().to(self.opt.device)
            train_class_weights_dict = {
                self.semeval_train.inv_label_ind[i]: self.train_class_weights[i] for i in np.arange(len(self.semeval_train.classes))}
            print(f'train_class_weights: {train_class_weights_dict}')
            # self.criterion = Loss(weight=self.train_class_weights_repri)
            self.criterion = Loss(pos_weight=torch.tensor(
                self.semeval_train.classes[1] / self.semeval_train.classes[0], dtype=torch.float))
            # TODO: Find a better way to assign the eps parameter for label smoothing.
            self.criterion.eps = self.opt.ls_eps

            self.eval_class_weights = [float(self.semeval_dev.classes[i]) / len(self.semeval_dev)
                                       for i in np.arange(len(self.semeval_dev.classes))]
            self.eval_class_weights_repri = torch.FloatTensor(
                self.eval_class_weights).reciprocal()
            eval_class_weights_dict = {
                self.semeval_dev.inv_label_ind[i]: self.eval_class_weights[i] for i in np.arange(len(self.semeval_dev.classes))}
            print(f'eval_class_weights: {eval_class_weights_dict}')
            # self.eval_criterion = Loss(weight=self.eval_class_weights_repri)
            self.eval_criterion = Loss(pos_weight=torch.tensor(
                self.semeval_dev.classes[1] / self.semeval_dev.classes[0], dtype=torch.float))
            # TODO: Find a better way to assign the eps parameter for label smoothing.
            self.eval_criterion.eps = self.opt.ls_eps

            self.eval_label_ind = self.semeval_dev.label_ind
            self.eval_inv_label_ind = self.semeval_dev.inv_label_ind
            self.eval_split = 'development'

        elif self.opt.mode == 'development':
            del self.semeval_train
            self.semeval_dev = MCL_WiC_Dataset(
                split='development',
                lang_1=opt.lang_1,
                lang_2=opt.lang_2,
                include_pos_tags=opt.include_pos_tags,
                batch_embedder=self.batch_embedder,
                k=opt.num_dependencies,
            )

            self.dev_data_loader = DataLoader(
                dataset=self.semeval_dev,
                # TODO: Check whether it could be increased further (dependent on computaitonal resource).
                batch_size=opt.batch_size * 4,
                shuffle=False,
                collate_fn=self.semeval_dev.collate_fn,
                pin_memory=True,
            )

            if self.opt.include_pos_tags:
                self.opt.num_pos_tag1 = len(self.semeval_dev.pos_tags2ind[1])
                self.opt.num_pos_tag2 = len(self.semeval_dev.pos_tags2ind[2])

            self.opt.num_label = len(self.semeval_dev.inv_label_ind)

            self.eval_class_weights = [float(self.semeval_dev.classes[i]) / len(self.semeval_dev)
                                       for i in np.arange(len(self.semeval_dev.classes))]

            self.eval_class_weights = [float(self.semeval_dev.classes[i]) / len(self.semeval_dev)
                                       for i in np.arange(len(self.semeval_dev.classes))]
            self.eval_class_weights_repri = torch.FloatTensor(
                self.eval_class_weights).reciprocal()
            eval_class_weights_dict = {
                self.semeval_dev.inv_label_ind[i]: self.eval_class_weights[i] for i in np.arange(len(self.semeval_dev.classes))}
            print(f'eval_class_weights: {eval_class_weights_dict}')
            # self.eval_criterion = Loss(weight=self.eval_class_weights_repri)
            self.eval_criterion = Loss(pos_weight=torch.tensor(
                self.semeval_dev.classes[1] / self.semeval_dev.classes[0], dtype=torch.float))
            # TODO: Find a better way to assign the eps parameter for label smoothing.
            self.eval_criterion.eps = self.opt.ls_eps

            self.eval_label_ind = self.semeval_dev.label_ind
            self.eval_inv_label_ind = self.semeval_dev.inv_label_ind
            self.eval_split = 'development'

        elif self.opt.mode == 'test':
            del self.semeval_train
            self.semeval_test = MCL_WiC_Dataset(
                split='test',
                lang_1=opt.lang_1,
                lang_2=opt.lang_2,
                include_pos_tags=opt.include_pos_tags,
                batch_embedder=self.batch_embedder,
                k=opt.num_dependencies,
            )

            self.test_data_loader = DataLoader(
                dataset=self.semeval_test,
                # TODO: Check whether it could be increased further (dependent on computaitonal resource).
                batch_size=opt.batch_size * 4,
                shuffle=False,
                collate_fn=self.semeval_test.collate_fn,
                pin_memory=True,
            )

            if self.opt.include_pos_tags:
                self.opt.num_pos_tag1 = len(self.semeval_test.pos_tags2ind[1])
                self.opt.num_pos_tag2 = len(self.semeval_test.pos_tags2ind[2])

            self.opt.num_label = len(self.semeval_test.inv_label_ind)

            self.eval_class_weights = [float(self.semeval_test.classes[i]) / len(self.semeval_test)
                                       for i in np.arange(len(self.semeval_test.classes))]

            self.eval_class_weights_repri = torch.FloatTensor(
                self.eval_class_weights).reciprocal()
            eval_class_weights_dict = {
                self.semeval_test.inv_label_ind[i]: self.eval_class_weights[i] for i in np.arange(len(self.semeval_test.classes))}
            print(f'eval_class_weights: {eval_class_weights_dict}')
            # self.eval_criterion = Loss(weight=self.test_class_weights_repri)
            self.eval_criterion = Loss(pos_weight=torch.tensor(
                self.semeval_dev.classes[1] / self.semeval_dev.classes[0], dtype=torch.float))
            # TODO: Find a better way to assign the eps parameter for label smoothing.
            self.eval_criterion.eps = self.opt.ls_eps

            self.eval_label_ind = self.semeval_test.label_ind
            self.eval_inv_label_ind = self.semeval_test.inv_label_ind
            self.eval_split = 'test'

        self.model = opt.model_class(opt).to(opt.device)
        self.model.train()
        # TODO: Check whether to separate optimizer for fine-tuning (i.e. different learning rate).
        # _params = filter(lambda p: p.requires_grad, chain(self.model.parameters(), self.batch_embedder.parameters()))
        # _params = filter(lambda p: all((p.requires_grad, p not in self.model.text_embeddings.parameters())), self.model.parameters())
        # _params = filter(
        #     lambda p: all(
        #         (
        #             p.requires_grad,
        #             not in_parameters(p, set(self.model.text_embeddings1.parameters()).union(
        #                 set(self.model.text_embeddings2.parameters()))),
        #         )), self.model.parameters())
        _params = filter(
            lambda p: all(
                (
                    p.requires_grad,
                    not in_parameters(p, self.model.text_embeddings.parameters()),
                )), self.model.parameters())
        self.optimizer = self.opt.optimizer(
            _params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if self.opt.fine_tuning:
            # _ft_params = filter(
            #     lambda p: p.requires_grad,
            #     set(self.model.text_embeddings1.parameters()).union(set(self.model.text_embeddings2.parameters()))
            # )
            _ft_params = filter(
                lambda p: p.requires_grad,
                self.model.text_embeddings.parameters(),
            )
            self.ft_optimizer = self.opt.optimizer(
                _ft_params, lr=self.opt.ft_learning_rate, weight_decay=self.opt.ft_l2reg)
        else:
            self.ft_optimizer = None

        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir()
            self.max_eval_acc = 0
            self.max_eval_f1 = 0
            self.global_step = 0
            self.epoch = 0
            self.repeat = 0
            self.continue_not_increase = 0
            self.accu_eval_acc = []
            self.accu_eval_f1 = []
            self._reset_params()
        else:
            files = [path.stem for path in self.ckpt_dir.glob('*.pt')]
            if len(files) > 0:
                steps = list(map(lambda x: x.split('_')[1], files))
                steps.sort(key=lambda x: int(x))
                most_recent_step = steps[-1]
                checkpoint = torch.load(
                    self.ckpt_dir / f'{self.opt.model_name}_{most_recent_step}.pt', map_location=self.opt.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                # NOTE: Currently the program does not support continue fine-tuning the BERT embeddings from checkpoints (assuming the fine-tuning process is only turned on for the first few epochs).
                # if self.opt.fine_tuning:
                #     self.ft_optimizer.load_state_dict(
                #         checkpoint['ft_optimizer_state_dict'])

                self.max_eval_acc = checkpoint['max_eval_acc']
                self.max_eval_f1 = checkpoint['max_eval_f1']
                self.global_step = checkpoint['global_step']
                self.epoch = checkpoint['epoch']
                self.continue_not_increase = checkpoint['continue_not_increase']
                print(f'Continue training from global step {self.global_step}')

                try:
                    self.repeat = checkpoint['repeat']
                    self.accu_eval_acc = checkpoint['accu_eval_acc']
                    self.accu_eval_f1 = checkpoint['accu_eval_f1']
                except KeyError:
                    print('This checkpoint is deprecated, please retrain the model.')
                    self.repeat = 0
                    self.accu_eval_acc = []
                    self.accu_eval_f1 = []
            else:
                self.max_eval_acc = 0
                self.max_eval_f1 = 0
                self.global_step = 0
                self.epoch = 0
                self.continue_not_increase = 0
                self.repeat = 0
                self.accu_eval_acc = []
                self.accu_eval_f1 = []
                self._reset_params()
        self._print_args()

        if torch.cuda.is_available():
            print('cuda memory allocated:',
                  torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        """Print out and store the trainable parameters.
        """
        n_trainable_params, n_nontrainable_params = 0, 0
        if self.opt.fine_tuning:
            params = self.model.parameters()
        else:
            # params = [p for p in self.model.parameters() if p not in self.model.text_embeddings.parameters()]
            # params = filter(lambda p: p not in self.model.text_embeddings.parameters(), self.model.parameters())
            # If fine-tuning is turned off, filter out the parameters from the BERT model.
            # params = filter(
            #     lambda p: not in_parameters(p, set(self.model.text_embeddings2.parameters()
            #                                        ).union(set(self.model.text_embeddings2.parameters()))),
            #     self.model.parameters())
            params = filter(
                lambda p: not in_parameters(p, self.model.text_embeddings.parameters()),
                self.model.parameters())
        for p in params:
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(f'n_trainable_params: {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}')

        self.pars_dir = self.log_dir / 'params'
        self.pars_dir.mkdir(exist_ok=True)

        with open(self.pars_dir / 'params.json', 'w') as json_file:
            json.dump(self.arg_dict, json_file, indent=2)

    def _reset_params(self):
        """Initialize the trainable parameters.
        """
        # for p in filter(lambda p: all((p.requires_grad, not in_parameters(p, set(self.model.text_embeddings1.parameters()).union(set(self.model.text_embeddings2.parameters()))))), self.model.parameters()):
        for p in filter(lambda p: all((p.requires_grad, not in_parameters(p, self.model.text_embeddings.parameters()))), self.model.parameters()):
            if len(p.shape) > 1:
                self.opt.initializer(p)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self):
        """Training loop of the model.
        """
        start_time = time.time()
        epoch = self.epoch
        while epoch < self.opt.num_epoch:
            print('>' * 100)
            print('epoch: ', epoch)
            increase_flag = False

            # tmp_int_time = time.time()

            for i_batch, sample_batched in enumerate(self.train_data_loader):

                # print('Load data: {}s'.format(time.time() - tmp_int_time))
                # tmp_int_time = time.time()

                self.global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                if self.opt.fine_tuning:
                    self.ft_optimizer.zero_grad()

                # print('Switch to training mode and zero optimizer: {}s'.format(time.time() - tmp_int_time))
                # tmp_int_time = time.time()

                # inputs = [sample_batched[col].to(
                #     self.opt.device) for col in self.opt.inputs_cols]
                inputs = sample_batched
                targets = sample_batched['label'].to(self.opt.device)

                outputs = self.model(
                    inputs, fine_tuning=epoch < self.opt.ft_epoch)

                # print('Network forward: {}s'.format(time.time() - tmp_int_time))
                # tmp_int_time = time.time()

                # loss = self.criterion(outputs, targets)
                if self.opt.label_smoothing:
                    loss_targets = torch.where(
                        torch.eq(targets, torch.zeros_like(targets, dtype=torch.long)),
                        torch.zeros_like(targets, dtype=torch.float) + self.opt.ls_eps,
                        torch.ones_like(targets, dtype=torch.float) - self.opt.ls_eps,
                    ).unsqueeze(dim=1)
                else:
                    loss_targets = targets.to(dtype=torch.float).unsqueeze(dim=1)
                loss = self.criterion(outputs, loss_targets)

                # print('Compute loss: {}s'.format(time.time() - tmp_int_time))
                # tmp_int_time = time.time()

                loss.backward()

                loss = loss.cpu()
                outputs = outputs.cpu()
                targets = targets.cpu()

                torch.cuda.empty_cache()

                # print('Compute gradients: {}s'.format(time.time() - tmp_int_time))
                # tmp_int_time = time.time()

                # NOTE: Clipping gradients.
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.5)
                self.optimizer.step()
                if all((self.opt.fine_tuning, epoch < self.opt.ft_epoch)):
                    # torch.nn.utils.clip_grad_norm_(self.batch_embedder.parameters(), 5)
                    self.ft_optimizer.step()

                # print('Update parameters: {}s'.format(time.time() - tmp_int_time))
                # tmp_int_time = time.time()

                if self.global_step % self.opt.log_step == 0:
                    eval_start_time = time.time()

                    with torch.no_grad():
                        targets = targets.numpy()
                        # output_labels = torch.argmax(outputs, -1).numpy()
                        output_labels = torch.round(torch.sigmoid(outputs)).numpy()
                        train_acc = accuracy_score(targets, output_labels)
                        train_f1 = f1_score(targets, output_labels)
                    # train_acc = n_correct / n_total

                    # print('Compute objective accuracy: {}s'.format(time.time() - tmp_int_time))
                    # tmp_int_time = time.time()

                    result_dict = self._evaluate()

                    # print('Evaluation of val set: {}s'.format(time.time() - tmp_int_time))
                    # tmp_int_time = time.time()

                    eval_loss = result_dict['eval_loss']
                    eval_acc = result_dict['eval_acc']
                    eval_f1 = result_dict['eval_f1']
                    self.writer.add_scalar(
                        'Loss/train', loss.item(), self.global_step)
                    self.writer.add_scalar(
                        'Loss/eval', eval_loss, self.global_step)
                    self.writer.add_scalar(
                        'Accuracy/train', train_acc, self.global_step)
                    self.writer.add_scalar(
                        'Accuracy/eval', eval_acc, self.global_step)
                    self.writer.add_scalar(
                        'F1Score/train', train_f1, self.global_step)
                    self.writer.add_scalar(
                        'F1Score/eval', eval_f1, self.global_step)

                    # print('Logging: {}s'.format(time.time() - tmp_int_time))
                    # tmp_int_time = time.time()
                    ckpt_dict = {
                        'global_step': self.global_step,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'max_eval_acc': self.max_eval_acc,
                        'max_eval_f1': self.max_eval_f1,
                        'continue_not_increase': self.continue_not_increase,
                        'repeat': self.repeat,
                        'accu_eval_acc': self.accu_eval_acc,
                        'accu_eval_f1': self.accu_eval_f1,
                    }
                    latest_ckpt = self.ckpt_dir / \
                        f'{self.opt.model_name}_{self.global_step}.pt'
                    if self.opt.save:
                        torch.save(ckpt_dict, latest_ckpt)

                    for ckpt in self.ckpt_dir.glob('*'):
                        if ckpt != latest_ckpt:
                            ckpt.unlink()

                    if eval_acc > self.max_eval_acc:
                        increase_flag = True
                        self.max_eval_acc = eval_acc
                        self.max_eval_f1 = eval_f1
                        if self.opt.save:
                            torch.save(self.model.state_dict(
                            ), self.best_model_dir / f'{self.opt.model_name}.pt')
                            print('>>> best model saved.')

                    print(f'step: {self.global_step}, train_loss: {loss.item():.4f}, eval_loss: {eval_loss:.4f}, train_acc: {train_acc:.4f}, eval_acc: {eval_acc:.4f}, train_f1: {train_f1:.4f}, eval_f1: {eval_f1:.4f}, evaluation time: {time.time() - eval_start_time:.4f}s, time elapsed: {time.time() - start_time:.4f}s')
                    # print('Saving checkpoint: {}s'.format(time.time() - tmp_int_time))
                    # tmp_int_time = time.time()

                    if all([epoch == self.opt.ft_epoch, self.opt.fine_tuning, self.opt.save]):
                        # torch.save(self.model.text_embeddings.state_dict(), self.ft_dir / 'pytorch_model.bin')
                        # self.model.text_embeddings.config.to_json_file(self.ft_dir / 'config.json')
                        # self.batch_embedder.tokenizer.save_vocabulary(self.ft_dir / 'vocab.txt')
                        self.model.text_embeddings.save_pretrained(self.ft_dir)
                        self.batch_embedder.tokenizer.save_pretrained(
                            self.ft_dir)

            if increase_flag is False:
                self.continue_not_increase += 1
                if self.continue_not_increase >= self.opt.early_stop:
                    print('early stop.')
                    break
            else:
                self.continue_not_increase = 0

            # torch.cuda.empty_cache()
            # del inputs, targets

            epoch += 1

        ckpt_dict = {
            'global_step': self.global_step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'max_eval_acc': self.max_eval_acc,
            'max_eval_f1': self.max_eval_f1,
            'continue_not_increase': self.continue_not_increase,
            'repeat': self.repeat,
            'accu_eval_acc': self.accu_eval_acc,
            'accu_eval_f1': self.accu_eval_f1,
        }
        if self.opt.save:
            torch.save(ckpt_dict, self.ckpt_dir / f'{self.opt.model_name}_{self.global_step}.pt')
        print(f'Finished training with {self.global_step} steps in {time.time() - start_time:.4f}s.')

    def _evaluate(self) -> dict:
        """Evaluate the model and return the result dictionary.

        Returns:
            dict: Dictionary of the model performance.
        """
        self.model.eval()

        # t_targets_all, t_outputs_all = None, None
        t_targets_all, t_outputs_all = [], []
        results = defaultdict(list)
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.dev_data_loader):
                t_inputs = t_sample_batched
                t_targets = t_sample_batched['label']
                t_outputs = self.model(t_inputs, fine_tuning=False).cpu()

                torch.cuda.empty_cache()
                t_targets_all.append(t_targets)
                t_outputs_all.append(t_outputs)

                # if t_targets_all is None:
                #     t_targets_all = t_targets
                #     t_outputs_all = t_outputs
                # else:
                #     t_targets_all = torch.cat(
                #         (t_targets_all, t_targets), dim=0)
                #     t_outputs_all = torch.cat(
                #         (t_outputs_all, t_outputs), dim=0)
            t_outputs_all = torch.cat(t_outputs_all, dim=0)
            t_targets_all = torch.cat(t_targets_all, dim=0)
            # loss = self.eval_criterion(t_outputs_all, t_targets_all)
            loss = self.eval_criterion(t_outputs_all, t_targets_all.to(dtype=torch.float).unsqueeze(dim=1))

        # t_targets_all = t_targets_all.numpy()
        # t_outputs_all = torch.argmax(t_outputs_all, dim=-1).numpy()
        t_outputs_all = torch.round(torch.sigmoid(t_outputs_all)).numpy()
        results['eval_acc'] = accuracy_score(t_targets_all, t_outputs_all)
        results['eval_f1'] = f1_score(t_targets_all, t_outputs_all)
        results['eval_loss'] = loss.item()

        return results

    def run(self):
        """Run the program as defined by the CLI arugments.
        """
        self.writer = SummaryWriter(log_dir=self.log_dir, filename_suffix=f'{self.opt.model_name}')
        # TODO: Maybe include model graph in the loggers.
        # for example_input in self.train_data_loader:
        #     break
        # self.writer.add_graph(self.model, example_input)
        if self.opt.mode == 'training':

            summary_keys = [
                'hparam/FinalAccuracy',
                'hparam/FinalF1Score',
                'hparam/BestAccuracy',
                'hparam/BestF1Score',
            ]

            # Start of training repeats loop.
            repeat = self.repeat
            while repeat < self.opt.repeats:
                print('-' * 100)
                print('repeat: ', repeat)

                self._train()
                print(f'max_eval_acc: {self.max_eval_acc}, max_eval_f1: {self.max_eval_f1}')
                print('#' * 100)
                print('Evaluate training results for the current parameter settings.')
                print('Evaluate on the final trained model:')
                result_dict = self._evaluate()
                self.accu_eval_acc.append(result_dict['eval_acc'])
                self.accu_eval_f1.append(result_dict['eval_f1'])
                repeat += 1

            # End of training repeats loop

            if self.opt.save:
                print('Evaluate on the best model:')
                self.model.load_state_dict(torch.load(
                    self.best_model_dir / f'{self.opt.model_name}.pt', map_location=self.opt.device))
                result_dict = self._evaluate()
                best_accu_eval_acc = [result_dict['eval_acc']]
                best_accu_eval_f1 = [result_dict['eval_f1']]

                stacked_lists = [
                    self.accu_eval_acc,
                    self.accu_eval_f1,
                    best_accu_eval_acc,
                    best_accu_eval_f1,
                ]
            else:
                stacked_lists = [
                    self.accu_eval_acc,
                    self.accu_eval_f1,
                    [0.0 for _ in np.arange(len(self.accu_eval_acc))],
                    [0.0 for _ in np.arange(len(self.accu_eval_f1))],
                ]

            stacked_stats = np.stack(stacked_lists, axis=0)
            summary_dict = compute_summary(summary_keys, stacked_stats)
            print(f'Performance Summary:\n{summary_dict}')

            self.writer.add_hparams(self.arg_dict, summary_dict)
            if wandb:
                wandb.run.summary.update(summary_dict)
                wandb.log(summary_dict)

            self.writer.close()
        elif any([self.opt.mode == 'development', self.opt.mode == 'test']):
            self.global_step = -1
            if self.opt.model_dir is None:
                raise ValueError(
                    'Please provide the directory of the saved state_dict.')
            state_dict = torch.load(
                self.opt.model_dir, map_location=self.opt.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)

            print(f'Evaluate on the loaded model ({self.opt.model_dir}):')
            result_dict = self._evaluate()

            with open(self.log_dir / f'{self.opt.mode}.json', 'w', encoding='utf-8') as json_file:
                json.dump(result_dict, json_file, indent=2)

            self.writer.add_hparams(self.arg_dict, result_dict)
            if wandb:
                wandb.run.summary.update(result_dict)
                wandb.log(result_dict)

            self.writer.close()
        else:
            raise ValueError('Unknown mode.')


def in_parameters(param: torch.Tensor, parameters: Generator) -> bool:
    """Check if the parameter is part of the module parameters.

    Args:
        param (torch.Tensor):
        parameters (Generator):

    Returns:
        bool: Flag to indicate whether param is part of parameters.
    """
    for p in parameters:
        if param is p:
            return True
    return False


def harmonic_mean(a: float, b: float) -> float:
    """Compute the harmonic mean of a and b.

    Args:
        a (float):
        b (float):

    Returns:
        float: The computed harmonic mean of a and b.
    """
    if a + b == 0:
        return 0.0
    return 2 * (a * b) / (a + b)


def compute_summary(keys: list, stacked_stats: np.ndarray) -> dict:
    """Compute the mean and std for the statistic and summarize in a dictionary given the keys.

    Args:
        keys (list): List of statistic measure names.
        stacked_stats (np.ndarray): Multiple results for the statistic measures. 2D tensor (number of measures, number of repeats).

    Returns:
        dict: Dictionary storing the mean and std of the statistics.
    """
    result_dict = {}
    means = np.mean(stacked_stats, axis=-1)
    stds = np.std(stacked_stats, axis=-1)
    for i_key, key in enumerate(keys):
        result_dict[key + '-mean'] = means[i_key]
        result_dict[key + '-std'] = stds[i_key]
    return result_dict
