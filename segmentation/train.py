
import numpy as np
import matplotlib.pyplot as plt

import shutil
import argparse
import os
import json
import random
import warnings
from termcolor import colored
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter

import imgaug # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa

import misc
import dataset
from net import DenseNet
from config import Config

####
class Trainer(Config):
    ####
    def view_dataset(self, mode='train'):
        train_pairs, valid_pairs = dataset.prepare_data_test()
        if mode == 'train':
            train_augmentors = self.train_augmentors()
            ds = dataset.DatasetSerial(train_pairs,
                            shape_augs=iaa.Sequential(train_augmentors[0]),
                            input_augs=iaa.Sequential(train_augmentors[1]))
        else:
            infer_augmentors = self.infer_augmentors()
            ds = dataset.DatasetSerial(valid_pairs,
                            shape_augs=iaa.Sequential(infer_augmentors))
        dataset.visualize(ds, 1)
        return
    ####
    def train_step(self, net, batch, optimizer, device):
        net.train() # train mode

        imgs, true = batch # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW



        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long() # not one-hot

        # -----------------------------------------------------------
        net.zero_grad() # not rnn so not accumulate

        logit = net(imgs) # forward
        prob = F.softmax(logit, dim=1)

        # has built-int log softmax so accept logit
        # true = torch.squeeze(true)
        loss = F.cross_entropy(logit, true, reduction='mean')

        prob = prob.permute(0, 2, 3, 1) # to NHWC
        pred = torch.argmax(prob, dim=-1)

        # with ignore index at 0
        foc = (true > 0).type(torch.float32)
        acc = (pred == true).type(torch.float32) * foc
        acc = torch.sum(acc) / torch.sum(foc)

        # gradient update
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------
        return dict(loss=loss.item(), 
                    acc=acc.item())
    ####
    def infer_step(self, net, batch, device):
        net.eval() # infer mode

        imgs, true = batch # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()

        # -----------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            logit = net(imgs) # forward
            prob = nn.functional.softmax(logit, dim=1)
            prob = prob.permute(0, 2, 3, 1) # to NHWC
            return dict(prob=prob.cpu().numpy(), 
                        true=true.numpy())
    ####
    def run_once(self):
        
        log_dir = self.log_dir

        misc.check_manual_seed(self.seed)
        train_pairs, valid_pairs = dataset.prepare_data_CANCER()
        print(len(train_pairs))
        # --------------------------- Dataloader

        train_augmentors = self.train_augmentors()
        train_dataset = dataset.DatasetSerial(train_pairs[:],
                        shape_augs=iaa.Sequential(train_augmentors[0]),
                        input_augs=iaa.Sequential(train_augmentors[1]))

        infer_augmentors = self.infer_augmentors()
        infer_dataset = dataset.DatasetSerial(valid_pairs[:],
                        shape_augs=iaa.Sequential(infer_augmentors))

        train_loader = data.DataLoader(train_dataset, 
                                num_workers=self.nr_procs_train, 
                                batch_size=self.train_batch_size, 
                                shuffle=True, drop_last=True)

        valid_loader = data.DataLoader(infer_dataset, 
                                num_workers=self.nr_procs_valid, 
                                batch_size=self.infer_batch_size, 
                                shuffle=True, drop_last=False)

        # --------------------------- Training Sequence

        if self.logging:
            misc.check_log_dir(log_dir)

        device = 'cuda'

        # networks
        input_chs = 3    
        net = DenseNet(input_chs, self.nr_classes)
        net = torch.nn.DataParallel(net).to(device)
        # print(net)

        # optimizers
        optimizer = optim.Adam(net.parameters(), lr=self.init_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.lr_steps)

        # load pre-trained models
        if self.load_network:
            saved_state = torch.load(self.save_net_path)

            new_saved_state = {}
            for key, value in saved_state.items():
                new_saved_state[key[7:]] = value
            net.load_state_dict(new_saved_state)
        #
        trainer = Engine(lambda engine, batch: self.train_step(net, batch, optimizer, 'cuda'))
        inferer = Engine(lambda engine, batch: self.infer_step(net, batch, 'cuda'))

        train_output = ['loss', 'acc']
        infer_output = ['prob', 'true']
        ##

        if self.logging:
            checkpoint_handler = ModelCheckpoint(log_dir, self.chkpts_prefix, 
                                            save_interval=1, n_saved=120, require_empty=False)
            # adding handlers using `trainer.add_event_handler` method API
            trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                                    to_save={'net': net}) 

        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(inferer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        RunningAverage(alpha=0.95, output_transform=lambda x: x['loss']).attach(trainer, 'loss')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['acc']).attach(trainer, 'acc')

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=['loss'])
        pbar.attach(inferer)

        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
                engine.terminate()
                warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
                checkpoint_handler(engine, {'net_exception': net})
            else:
                raise e

        # writer for tensorboard logging
        if self.logging:
            writer = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file) # create empty file

        @trainer.on(Events.EPOCH_STARTED)
        def log_lrs(engine):
            if self.logging:
                lr = float(optimizer.param_groups[0]['lr'])
                writer.add_scalar("lr", lr, engine.state.epoch)
            # advance scheduler clock
            scheduler.step()

        ####
        def update_logs(output, epoch, prefix, color):
            # print values and convert
            max_length = len(max(output.keys(), key=len))
            for metric in output:
                key = colored(prefix + '-' + metric.ljust(max_length), color)
                print('------%s : ' % key, end='')
                print('%0.7f' % output[metric])
            if 'train' in prefix:
                lr = float(optimizer.param_groups[0]['lr'])
                key = colored(prefix + '-' + 'lr'.ljust(max_length), color)
                print('------%s : %0.7f' % (key, lr))

            if not self.logging:
                return

            # create stat dicts
            stat_dict = {}
            for metric in output:
                metric_value = output[metric] 
                stat_dict['%s-%s' % (prefix, metric)] = metric_value

            # json stat log file, update and overwrite
            with open(json_log_file) as json_file:
                json_data = json.load(json_file)

            current_epoch = str(epoch)
            if current_epoch in json_data:
                old_stat_dict = json_data[current_epoch]
                stat_dict.update(old_stat_dict)
            current_epoch_dict = {current_epoch : stat_dict}
            json_data.update(current_epoch_dict)

            with open(json_log_file, 'w') as json_file:
                json.dump(json_data, json_file)

            # log values to tensorboard
            for metric in output:
                writer.add_scalar(prefix + '-' + metric, output[metric], current_epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_running_results(engine):
            """
            running training measurement
            """
            training_ema_output = engine.state.metrics #
            update_logs(training_ema_output, engine.state.epoch, prefix='train-ema', color='green')

        ####
        def get_init_accumulator(output_names):
            return {metric : [] for metric in output_names}

        import cv2
        def process_accumulated_output(output):
            def uneven_seq_to_np(seq, batch_size=self.infer_batch_size):
                if self.infer_batch_size == 1:
                    return np.squeeze(seq)
                    
                item_count = batch_size * (len(seq) - 1) + len(seq[-1])
                cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
                for idx in range(0, len(seq)-1):
                    cat_array[idx   * batch_size : 
                            (idx+1) * batch_size] = seq[idx] 
                cat_array[(idx+1) * batch_size:] = seq[-1]
                return cat_array
            #
            prob = uneven_seq_to_np(output['prob'])
            true = uneven_seq_to_np(output['true'])

            # cmap = plt.get_cmap('jet')
            # epi = prob[...,1]
            # epi = (cmap(epi) * 255.0).astype('uint8')
            # cv2.imwrite('sample.png', cv2.cvtColor(epi, cv2.COLOR_RGB2BGR))

            pred = np.argmax(prob, axis=-1)
            true = np.squeeze(true)

            # deal with ignore index
            pred = pred.flatten()
            true = true.flatten()
            pred = pred[true != 0] - 1
            true = true[true != 0] - 1

            acc = np.mean(pred == true)
            inter = (pred * true).sum()
            total = (pred + true).sum()
            dice = 2 * inter / total
            #
            proc_output = dict(acc=acc, dice=dice)
            return proc_output

        @trainer.on(Events.EPOCH_COMPLETED)
        def infer_valid(engine):
            """
            inference measurement
            """
            inferer.accumulator = get_init_accumulator(infer_output)
            inferer.run(valid_loader)
            output_stat = process_accumulated_output(inferer.accumulator)
            update_logs(output_stat, engine.state.epoch, prefix='valid', color='red')

        @inferer.on(Events.ITERATION_COMPLETED)
        def accumulate_outputs(engine):
            batch_output = engine.state.output
            for key, item in batch_output.items():
                engine.accumulator[key].extend([item])
        ###
        #Setup is done. Now let's run the training
        trainer.run(train_loader, self.nr_epochs)
        return
    ####

####
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    trainer = Trainer()
    trainer.run_once()
    # trainer.view_dataset()