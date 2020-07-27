from copy import deepcopy

import torch
import time
import math
import os
import sys
import re
import shutil
import glob
import csv
import yaml
import operator
import logging
import numpy as np
try:
    import apex
    has_apex = True
except ImportError:
    has_apex = False
from collections import OrderedDict

from torch import distributed as dist
import torch.nn as nn


def get_state_dict(model):
    if isinstance(model, ModelEma):
        return get_state_dict(model.ema)
    else:
        new_state_dict = OrderedDict()
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v.cpu()
        return new_state_dict


class CheckpointSaver:
    def __init__(
            self,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10000):

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, model, optimizer, args, stage, epoch, step=None, model_pool=None, potential=None, model_ema=None, metric=None):
        assert stage >= 0
        assert epoch >= 0
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            if step is not None:
                filename = '{}-{}-{}-{}'.format(self.save_prefix, stage,
                                             epoch, step) + self.extension
            else:
                filename = '{}-{}-{}'.format(self.save_prefix, stage, epoch) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            self._save(save_path, model, optimizer, args, stage, epoch, model_ema, metric)
            if model_pool is not None:
                model_filename = 'modelpool-{}.yaml'.format(stage)
                model_save_path = os.path.join(self.checkpoint_dir, model_filename)
                self._save_yaml(model_save_path, model_pool)
            if potential is not None:
                model_filename = 'potential-{}.yaml'.format(stage)
                model_save_path = os.path.join(self.checkpoint_dir, model_filename)
                self._save_yaml(model_save_path, potential)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            logging.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                shutil.copyfile(save_path, os.path.join(self.checkpoint_dir, 'model_best' + self.extension))

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, model, optimizer, args, stage, epoch, model_ema=None, metric=None):
        save_state = {
            'stage': stage,
            'epoch': epoch,
            'state_dict': get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'version': 2,  # version < 2 increments epoch before save
        }
        if model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(model_ema)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                logging.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                logging.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, model, optimizer, args, epoch, model_ema=None, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema)
        if os.path.exists(self.last_recovery_file):
            try:
                logging.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                logging.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''
    def _save_yaml(self, save_path, model_pool):
        with open(save_path, 'w', encoding='utf8') as f:
            yaml.safe_dump(model_pool, f, default_flow_style=False, allow_unicode=True)



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]


'''
def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir
'''


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


class ModelEma:
    """ Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            logging.info("Loaded state_dict_ema")
        else:
            logging.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(outdir, args):
    default_level = logging.INFO
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=default_level,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    if args.local_rank == 0:
        fh = logging.FileHandler(os.path.join(outdir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)


def get_outdir(*path, scripts_to_save=None):
    outdir = os.path.join(*path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if scripts_to_save == 'All':
        targetdir = os.path.join(outdir, 'scripts')
        shutil.copytree('./', targetdir,
                        ignore=shutil.ignore_patterns('output*', 'cache*', 'results*', 'convert*', 'notebooks*',
                                                      'dataset*'))
    elif scripts_to_save is not None:
        os.mkdir(os.path.join(outdir, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(outdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    return outdir


class BN_Correction(object):
    """docstring for BN_Correction"""
    def __init__(self, model, train_data, args):
        super(BN_Correction, self).__init__()
        self.model = model
        num_bn = 0

        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or (has_apex and isinstance(layer, apex.parallel.SyncBatchNorm)):
                layer.reset_running_stats()
                layer.momentum = 1
                num_bn += 1
        if args.local_rank == 0:
            print('Number of BN momentumn reset: {}'.format(num_bn))
        # raw_data = []
        # count = 0
        # data = train_data
        # self.data = torch.cat(raw_data, dim=0)
        # assert h == w == img_dim, (h, w, img_dim)
        self.data = train_data
        self.data = self.data.cuda()

        # self.save is deleted in this script

    def __call__(self, encoding=None, start_block=None, end_block=None, reverse_encod=False, net="teacher"):
        # net_idx = str(self.model.net_id)
        # if self.logger:
        #   self.logger.info("Reseting BN running stats in Net_idx: {}".format(net_idx))
        torch.cuda.synchronize()
        tic = time.time()
        self.model.train()
        with torch.no_grad():
            if net=='supernet':
                output = self.model(self.data,
                               encoding=encoding,
                               start_block=start_block,
                               end_block=end_block,
                               reverse_encod=reverse_encod)
            else:
                output = self.model(self.data,
                               end_block=end_block)
        torch.cuda.synchronize()
        toc = time.time()
        self.model.eval()

        # print("Reset BN running stats cost time:{:.1f} s".format(toc-tic))
        return output
