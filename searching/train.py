import argparse

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

from timm_.data import Dataset, create_loader, resolve_data_config
from timm_.models import create_model, resume_checkpoint
from timm_.utils import *
from timm_.loss import *
from timm_.optim import create_optimizer
from timm_.scheduler import create_scheduler
from dna.distill_train import distill_train
from dna.student_supernet import StudentSuperNet
from initialize import Initial

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_config', default=None, type=str)
parser.add_argument('--hyperparam_config', default=None, type=str)
parser.add_argument('--pretrain', action='store_true' , default=False, help='To save model as a pretrained model')
parser.add_argument('--init_classifier', action='store_true' , default=False, help='To finetuen the classifier')
parser.add_argument('--exp_dir', default='', type=str)
parser.add_argument('--index', default='', type=str)
parser.add_argument("--local_rank", default=0, type=int)

def main():
# ================== Init =================
    args = parser.parse_args()
    initial = Initial(args,
                      base_configs=['train_pipeline.yaml',
                                          'data.yaml'],
                      hyperparam_config=args.hyperparam_config)
    args = initial.args
# ---- Set Output Dir & Logger ----
    output_dir = ''
    if args.local_rank == 0:
        # print(args.__dict__)
        output_base = args.output if args.output else './output'
        if args.index:
            exp_name = '{}-{}-{}-ep{}-lr{}-bs{}-{}'.format(args.index, args.opt, args.sched, args.step_epochs,
            args.lr[0] if isinstance(args.lr, list) else args.lr,
            args.batch_size, time.strftime("%Y%m%d-%H%M%S"))
        else:
            exp_name = '{}-{}-ep{}-lr{}-bs{}-{}'.format(args.opt, args.sched, args.step_epochs,
            args.lr[0] if isinstance(args.lr, list) else args.lr,
            args.batch_size, time.strftime("%Y%m%d-%H%M%S"))

        output_dir = get_outdir(output_base, 'test', args.exp_dir, exp_name, scripts_to_save='All')
    setup_default_logging(output_dir, args)
    writer = None
    if args.local_rank == 0:
        writer = SummaryWriter(output_dir)
# ---- Init ----
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        if args.local_rank == 0:
            logging.info('WORLD_SIZE in os.environ is {}'.format(int(os.environ['WORLD_SIZE'])))
            logging.info(args)
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            logging.warning(
                'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23334', rank=args.local_rank,
                                            world_size=int(os.environ['WORLD_SIZE']))
        # torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        assert args.rank >= 0
        logging.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. CUDA %d, Process %d, total %d.'
            % (args.local_rank, args.rank, args.world_size))
    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)
# ================= Load Model ==================
    if args.test_dispatch.lower() == 'b0':
        encoding = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
        block_layers_num = [2, 2, 3, 3, 4, 1]
        supernet = StudentSuperNet.dispatch(num_classes=1000,
                                            init_op_list=encoding,
                                            block_layers_num=block_layers_num)
        # efficient b0
    elif args.test_dispatch.lower() == 'b7':
        encoding = [0,0,0,0,0,0,0,
                    1,1,1,1,1,1,1,
                    0,0,0,0,0,0,0,0,0,0,
                    1,1,1,1,1,1,1,1,1,1,
                    1,1,1,1,1,1,1,1,1,1,1,1,1,
                    0,0,0,0]
        block_layers_num = [7,7,10,10,13,4]
        supernet = StudentSuperNet.dispatch(num_classes=1000,
                                            init_op_list=encoding,
                                            block_layers_num=block_layers_num)
        # Efficient B7
    elif args.test_dispatch.lower() == 'crsupernet':
        block_layers_num = [2,2,2,4,6,2]
        supernet = StudentSuperNet(num_classes=1000,
                                   block_layers_num=block_layers_num)
        # Computation Reallocation Supernet
    else:
        supernet = StudentSuperNet(num_classes=args.num_classes)
    teacher = create_model('tf_efficientnet_b7',
                           pretrained=True,
                           num_classes=args.num_classes)

    # print("model 0 n_params:", get_model_parameters_number(supernet))
    #
    # exit(0)
    data_config = resolve_data_config(vars(args), verbose=args.local_rank == 0)

    # optionally resume from a checkpoint
    optimizer_state = None
    resume_epoch = None
    resume_stage = None
    model_pool = None
    if args.resume:
        optimizer_state, resume_epoch, resume_stage = resume_checkpoint(supernet, args.resume)
        if args.model_pool:
            with open(args.model_pool, 'r') as f:
                model_pool = yaml.safe_load(f)


    supernet.cuda()
    teacher.cuda()
# ---- Create Optimizer ----
    optimizer = create_optimizer(args, supernet)
    # if optimizer_state is not None:
    #     optimizer.load_state_dict(optimizer_state)

    use_amp = False
    if has_apex and args.amp:
        supernet, optimizer = amp.initialize(supernet,
                                             optimizer,
                                             opt_level='O1')
        use_amp = True
    if args.local_rank == 0:
        logging.info('NVIDIA APEX {}. AMP {}.'.format(
            'installed' if has_apex else 'not installed',
            'on' if use_amp else 'off'))

    if args.distributed:
        if args.sync_bn:
            try:
                if has_apex:
                    supernet = convert_syncbn_model(supernet)
                    teacher = convert_syncbn_model(teacher)
                else:
                    supernet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(supernet)
                    teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
                if args.local_rank == 0:
                    logging.info('Converted model to use Synchronized BatchNorm.')
            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex:
            supernet = DDP(supernet, delay_allreduce=True)
            teacher = DDP(teacher, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            supernet = DDP(supernet, device_ids=[args.local_rank],
                           find_unused_parameters=True)  # can use device str in Torch >= 1.1
            teacher = DDP(teacher, device_ids=[args.local_rank],
                           find_unused_parameters=True)
        # NOTE: EMA model does not need to be wrapped by DDP
# ---- Create Scheduler ----
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    start_stage = None
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    # if start_epoch > 0:
    #     lr_scheduler.step(start_epoch)
    if args.start_stage is not None:
        # a specified start_epoch will always override the resume epoch
        start_stage = args.start_stage
    elif resume_stage is not None:
        start_stage = resume_stage

    # if args.local_rank == 0:
    #     logging.info('Scheduled epochs: {}'.format(num_epochs))
# ---- Create Data Loader ----
    train_dir = os.path.join(args.datadir, 'train')
    dataset_train = Dataset(train_dir)
    eval_dir = os.path.join(args.datadir, 'validation')
    dataset_eval = Dataset(eval_dir)
    loader_train = create_loader(dataset_train,
                                 input_size=data_config['input_size'],
                                 batch_size=args.batch_size,
                                 is_training=True,
                                 use_prefetcher=args.prefetcher,
                                 rand_erase_prob=args.reprob,
                                 rand_erase_mode=args.remode,
                                 color_jitter=args.color_jitter,
                                 interpolation='random',
                                 mean=data_config['mean'],
                                 std=data_config['std'],
                                 num_workers=args.workers,
                                 distributed=args.distributed)
    loader_eval = create_loader(dataset_eval,
                                input_size=data_config['input_size'],
                                batch_size=args.batch_size,
                                is_training=False,
                                use_prefetcher=args.prefetcher,
                                interpolation=data_config['interpolation'],
                                mean=data_config['mean'],
                                std=data_config['std'],
                                num_workers=args.workers,
                                distributed=args.distributed)
# ---- Loss Function ----
    target_loss_fn = nn.CrossEntropyLoss().cuda()
    klcos_loss_fn = KLCosineSimilarity()
    mse_loss_fn = nn.MSELoss()
    rkd_a_loss_fn = RKDAngle()
    rkd_d_loss_fn = RKDDistance()
    att_loss_fn = AttentionTransfer()
    if args.guide_loss_fn.lower() == 'klcos':
        guide_loss_fn = klcos_loss_fn.cuda()
    elif args.guide_loss_fn.lower() == 'mse':
        guide_loss_fn = mse_loss_fn.cuda()
    else:
        raise NotImplementedError
# ---- Saver ----
    saver = None
    eval_metric = 'loss'
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        # print(teacher)
# ================ Train & Eval & Save ===================
    distill_train(supernet=supernet,
                  teacher=teacher,
                  train_loader=loader_train,
                  eval_loader=loader_eval,
                  poten_loader=loader_eval,
                  optimizer=optimizer,
                  scheduler=lr_scheduler,
                  guide_loss_fn=guide_loss_fn,
                  target_loss_fn=target_loss_fn,
                  saver=saver,
                  args=args,
                  start_stage=start_stage,
                  start_epoch=start_epoch,
                  model_pool=model_pool,
                  writer=writer)



if __name__ == '__main__':
    main()
