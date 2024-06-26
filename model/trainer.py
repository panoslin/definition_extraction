"""
A trainer class.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchcrf import CRF

from model.gcn import GCNClassifier
from utils import torch_utils

random.seed(1234)


class GCNTrainer:
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        self.crf = CRF(self.opt['num_class'], batch_first=True)
        self.bc = nn.BCELoss()
        if torch.cuda.is_available():
            self.model.cuda()
            self.criterion.cuda()
            self.crf.cuda()
            self.bc.cuda()

        self.optimizer = torch_utils.get_optimizer(
            opt['optim'],
            self.parameters,
            opt['lr']
        )

    def update(self, batch):
        inputs, labels, sent_labels, dep_path, tokens, head, lens = self.unpack_batch(batch, self.opt['cuda'])

        _, _, _, _, terms, _, _ = inputs

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, class_logits, selections, term_def, not_term_def, term_selections = self.model(inputs)

        labels = labels - 1
        labels[labels < 0] = 0
        mask = inputs[1].float()
        mask[mask == 0.] = -1.
        mask[mask == 1.] = 0.
        mask[mask == -1.] = 1.
        mask = mask.byte()
        loss = -self.crf(logits, labels, mask=mask)

        sent_loss = self.bc(class_logits, sent_labels)
        loss += self.opt['sent_loss'] * sent_loss

        selection_loss = self.bc(selections.view(-1, 1), dep_path.view(-1, 1))
        loss += self.opt['dep_path_loss'] * selection_loss

        term_def_loss = -self.opt['consistency_loss'] * (term_def - not_term_def)
        loss += term_def_loss
        # loss += self.opt['consistency_loss'] * not_term_def

        term_loss = self.opt['sent_loss'] * self.bc(term_selections.view(-1, 1), terms.float().view(-1, 1))
        loss += term_loss

        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val, sent_loss.item(), term_loss.item()

    def predict(self, batch, unsort=True):
        inputs, labels, sent_labels, dep_path, tokens, head, lens = self.unpack_batch(batch, self.opt['cuda'])

        orig_idx = batch[-1]
        # forward
        self.model.eval()
        logits, sent_logits, _, _, _, _ = self.model(inputs)

        labels = labels - 1
        labels[labels < 0] = 0
        mask = inputs[1].float()
        mask[mask == 0.] = -1.
        mask[mask == 1.] = 0.
        mask[mask == -1.] = 1.
        mask = mask.byte()
        loss = -self.crf(logits, labels, mask=mask)

        probs = F.softmax(logits, dim=1)
        predictions = self.crf.decode(logits, mask=mask)

        sent_predictions = sent_logits.round().long().data.cpu().numpy()

        if unsort:
            _, predictions, probs, sent_predictions = [list(t) for t in zip(*sorted(
                zip(orig_idx, predictions, probs, sent_predictions)))]
        return predictions, probs, loss.item(), sent_predictions

    def update_lr(self, new_lr):
        """
        This function updates the learning rate of the optimizer used by the Trainer.
        It sets a new learning rate new_lr for all parameter groups in the optimizer.

        The learning rate influences the speed and quality of the learning process
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def load(self, filename):
        """
        The load function loads a model from the given filename using torch.load.

        It then loads the state dictionary of the model and the configuration from the checkpoint.

        If an exception occurs during the loading process,
        it prints an error message and exits the program.
        """
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model':  self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    @staticmethod
    def unpack_batch(batch, cuda):
        if torch.cuda.is_available():
            inputs = [Variable(b.cuda()) for b in batch[:7]]
            labels = Variable(batch[7].cuda())
            sent_labels = Variable(batch[8].cuda())
            dep_path = Variable(batch[9].cuda())
        else:
            inputs = [Variable(b) for b in batch[:7]]
            labels = Variable(batch[7])
            sent_labels = Variable(batch[8])
            dep_path = Variable(batch[9])

        tokens = batch[0]
        head = batch[3]
        lens = batch[1].eq(0).long().sum(1).squeeze()
        return inputs, labels, sent_labels, dep_path, tokens, head, lens
