import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import copy
import tensorflow as tf


class Trainer:

    def __init__(self, source_train_data, source_val_data, loss_fun=nn.CrossEntropyLoss(), device='cuda', num_workers=None, tensorboard_writer=None, log=True, log_every=None):
        self.source_train_data = source_train_data
        self.source_val_data = source_val_data

        self.loss_fun = loss_fun
        self.device = device
        self.num_workers = cpu_count() if num_workers is None else num_workers
        self.tensorboard_writer = tensorboard_writer

        self.log_every = log_every
        self.log = log

    def train(self, model, optimizer, lr_scheduler, batch_size, epochs, val_batch_size=256, epoch_offset=0, iterations_per_epoch=None):
        source_train_dataloader = DataLoader(self.source_train_data, num_workers=self.num_workers, batch_size=batch_size, shuffle=True)
        source_val_dataloader = DataLoader(self.source_val_data, num_workers=self.num_workers, batch_size=val_batch_size, shuffle=True)

        best_val_acc = 0
        best_model = copy.deepcopy(model.state_dict())
        model.to(self.device)
        for epoch in range(epoch_offset, epochs):
            source_train_iter = iter(source_train_dataloader)
            total_iterations = len(source_train_dataloader)

            if iterations_per_epoch is not None:
                total_iterations = min(total_iterations, iterations_per_epoch)

            log_every = total_iterations // 3 if self.log_every is None else self.log_every

            self._train_epoch(model, optimizer, source_train_iter, total_iterations, epoch, log_every)

            if lr_scheduler is not None:
                lr_scheduler.step()

            source_val_iter = iter(source_val_dataloader)
            total_iterations = len(source_val_dataloader)

            val_source_acc = self._validate_epoch(model, source_val_iter, total_iterations, epoch)

            if val_source_acc >= 1:
                return
            if val_source_acc > best_val_acc:
                best_model = copy.deepcopy(model.state_dict())
                best_val_acc = val_source_acc

        model.load_state_dict(best_model)

    def _train_epoch(self, model, optimizer, source_train_iter, total_iterations, epoch, log_every):
        if self.log:
            print('Epoch %d' % epoch)

        model.train()
        train_loss = torch.tensor([0.0])
        train_source_acc = torch.tensor([0.0])

        for iteration in range(total_iterations):
            self._train_iteration(model, optimizer, log_every, iteration, total_iterations, epoch, source_train_iter, train_loss, train_source_acc)

        train_source_acc /= total_iterations
        train_loss /= total_iterations

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="losses/train/total_loss", simple_value=train_loss),
                tf.Summary.Value(tag="acc/train/source_acc", simple_value=train_source_acc),

            ]), epoch)

        if self.log:
            print('(train) acc: %.4f, loss: %.6f' % (train_source_acc, train_loss))

    def _train_iteration(self, model, optimizer, log_every, iteration, total_iterations, epoch, source_train_iter, train_loss, train_source_acc):
        optimizer.zero_grad()

        source_input, source_label = next(source_train_iter)
        source_logits = model(source_input.to(self.device))

        loss = self.loss_fun(source_logits, source_label.to(self.device))

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.cpu()
            train_source_acc += self._accuracy(source_logits, source_label)

        if self.log and log_every > 0 and iteration != 0 and iteration % log_every == 0:
            print('%d/%d %.6f' % (iteration, total_iterations, loss.cpu()))

    def _validate_epoch(self, model, source_val_iter, total_iterations, epoch):
        val_source_acc = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])

        with torch.no_grad():
            model.eval()

            for iteration in range(total_iterations):
                self._validate_iteration(model, source_val_iter, epoch, val_loss, val_source_acc)

        val_source_acc /= total_iterations
        val_loss /= total_iterations

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="losses/val/total_loss", simple_value=val_loss),
                tf.Summary.Value(tag="acc/val/source_acc", simple_value=val_source_acc),
            ]), epoch)

        if self.log:
            print('(val) acc: %.4f, loss: %.6f' % (val_source_acc, val_loss))

        return val_source_acc

    def _validate_iteration(self, model, source_val_iter, epoch, val_loss, val_source_acc):
        source_input, source_label = next(source_val_iter)
        source_logits = model(source_input.to(self.device))

        loss = self.loss_fun(source_logits, source_label.to(self.device))

        val_loss += loss.cpu()
        val_source_acc += self._accuracy(source_logits, source_label)

    def _accuracy(self, logits, label):
        if logits.shape[-1] == 1:
            return ((logits.cpu() >= 0) == label.byte()).float().mean()
        else:
            return (torch.argmax(logits.cpu(), 1) == label).float().mean()
