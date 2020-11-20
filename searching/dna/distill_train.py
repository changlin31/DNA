import itertools
import random

try:
    import apex
    has_apex = True
except ImportError:
    has_apex = False

from timm_.utils import *
from timm_.loss import *
from timm_.optim import create_optimizer
from timm_.scheduler import create_scheduler

import torch
import torch.nn as nn
import torchvision.utils


def distill_train(supernet,
                  teacher,
                  train_loader,
                  eval_loader,
                  poten_loader,
                  optimizer,
                  scheduler,
                  guide_loss_fn,
                  target_loss_fn=None,
                  saver=None,
                  args=None,
                  start_stage=None,
                  start_epoch=0,
                  model_pool=None,
                  writer=None):

    if args.reset_bn_eval:
        reset_data = next(iter(train_loader))
    else:
        reset_data = None
    if model_pool is not None and args.local_rank == 0:
        logging.info('loaded model pool:\n{}'.format(model_pool))
    model_pool, skip_first_train, start_stage, start_epoch \
        = _process_startpoint(args=args,
                              supernet=supernet,
                              start_stage=start_stage,
                              start_epoch=start_epoch,
                              model_pool=model_pool)

    for stage in range(start_stage, args.stage_num):
        if args.guide_input:
            model_pool = []
        loss = None
        eval_losses = []
        if stage != start_stage:
            start_epoch = 1
        if start_epoch == 1:
            if args.separate_train:
                optimizer = create_optimizer(args,
                                             model=supernet.module.get_block(stage) \
                                                 if hasattr(supernet, 'module') \
                                                 else supernet.get_block(stage),
                                             stage=stage)
            else:
                optimizer = create_optimizer(args, supernet, stage=stage)
            scheduler = create_scheduler(args, optimizer)[0]
        if args.reverse_train:
            stage = args.stage_num - 1 - stage
        if not skip_first_train and not args.eval_mode:
            loss = _train_stage(supernet=supernet,
                                teacher=teacher,
                                train_loader=train_loader,
                                eval_loader=eval_loader,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                loss_fn=guide_loss_fn,
                                saver=saver,
                                args=args,
                                stage=stage,
                                model_pool=model_pool,
                                start_epoch=start_epoch,
                                writer=writer)
        else:
            skip_first_train = False
        if not (args.guide_input and args.train_mode):
            # not skip search
            potential = None
            if len(args.test_dispatch) == 0 or args.test_dispatch == 'crsupernet':
                stage_model_pool = get_all_models(supernet=supernet,
                                                  stage=stage,
                                                  model_pool=model_pool,
                                                  args=args)

                potential = _potential(supernet=supernet,
                                       teacher=teacher,
                                       eval_loader=poten_loader,
                                       loss_fn=guide_loss_fn,
                                       args=args,
                                       stage=stage,
                                       stage_model_pool=stage_model_pool,
                                       reset_data=reset_data)
                eval_losses = potential['loss']
                stage_model_pool, eval_losses, top_index = get_best_items(
                    candidates=stage_model_pool,
                    value=eval_losses,
                    top_num=args.top_model_num,
                    best='top' if args.label_train else 'bottom', )
                model_pool = stage_model_pool.copy()

                if args.local_rank == 0:
                    potential['top_index'] = top_index
                    logging.info('Stage: {}  model pool:\n{}'.format(stage, model_pool))
                    logging.info('loss:{}'.format(eval_losses[:args.top_model_num]))

            if len(eval_losses) == 0 and loss is None:
                loss = 100.
            elif len(eval_losses) != 0:
                loss = eval_losses[0]

            if saver is not None:
                saver.save_checkpoint(supernet,
                                      optimizer,
                                      args,
                                      stage=stage,
                                      epoch=args.step_epochs + 1,
                                      model_pool=model_pool,
                                      potential=potential,
                                      metric=loss)

    # ---- last stage ----
    if not args.distill_last_stage:
        stage = args.stage_num
        if start_epoch == 1:
            optimizer = create_optimizer(args, supernet, stage=stage)
            scheduler = create_scheduler(args, optimizer)[0]
        eval_losses = []
        if not skip_first_train:
            loss = _train_stage(supernet=supernet,
                                teacher=teacher,
                                train_loader=train_loader,
                                eval_loader=eval_loader,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                loss_fn=target_loss_fn,
                                saver=saver,
                                args=args,
                                stage=stage,
                                model_pool=model_pool,
                                use_target=True,
                                writer=writer)
        if len(args.test_dispatch) == 0:
            stage_model_pool = get_all_models(supernet=supernet,
                                              stage=stage,
                                              model_pool=model_pool,
                                              args=args)
            potential = _potential(supernet=supernet,
                                   teacher=teacher,
                                   eval_loader=poten_loader,
                                   loss_fn=target_loss_fn,
                                   args=args,
                                   stage=stage,
                                   stage_model_pool=stage_model_pool,
                                   use_target=True,
                                   reset_data=reset_data)
            eval_losses = potential['prec1']
            stage_model_pool, eval_losses, top_index = get_best_items(
                candidates=stage_model_pool,
                value=eval_losses,
                top_num=args.top_model_num,
                best='top')
            model_pool = stage_model_pool.copy()

            if args.local_rank == 0:
                potential['top_index'] = top_index
                logging.info('Stage: {}  model pool:\n{}'.format(stage, model_pool))
                logging.info('loss:{}'.format(eval_losses[:args.top_model_num]))

        if saver is not None:
            saver.save_checkpoint(supernet,
                                  optimizer,
                                  args,
                                  stage=stage,
                                  epoch=args.step_epochs + 1,
                                  model_pool=model_pool,
                                  metric=loss)
            logging.info('Potential Dict:\n {}'.format(potential))

    if writer is not None:
        writer.close()
    return model_pool


def _process_startpoint(args,
                        supernet,
                        model_pool,
                        start_stage,
                        start_epoch,
                        ):
    if args.reverse_train and start_stage is not None:
        start_stage = args.stage_num - start_stage - 1
    if model_pool is None:
        model_pool = []
    skip_first_train = False
    if start_stage is None:
        start_stage = 0

    model_len = 0
    for i in range(start_stage + 1):
        if hasattr(supernet, 'module'):
            current_len = supernet.module.get_layers(i)
        else:
            current_len = supernet.get_layers(i)
        model_len += current_len
    if start_epoch >= args.step_epochs:
        if len(model_pool) == 0 and not args.guide_input:
            if start_stage == 0:
                skip_first_train = True
            else:
                raise RuntimeError
        else:
            if args.guide_input:
                if start_epoch > args.step_epochs + 1:
                    start_stage += 1
                else:
                    skip_first_train = True
            else:
                if len(model_pool[0]) < model_len:
                    skip_first_train = True
                elif len(model_pool[0]) == model_len:
                    start_stage += 1
                else:
                    raise RuntimeError
        start_epoch = 1
    else:
        if not args.guide_input:
            if len(model_pool) == 0:
                assert (model_len - current_len) == 0
            else:
                assert (len(model_pool[0]) == (model_len - current_len))

    if hasattr(supernet, 'module'):
        assert (args.stage_num == len(supernet.module.block_cfgs))
    else:
        assert (args.stage_num == len(supernet.block_cfgs))
    assert (start_stage <= args.stage_num)
    if not args.distill_last_stage:
        args.stage_num -= 1
        if start_stage > args.stage_num:
            start_stage = args.stage_num
    if start_epoch == 0:
        start_epoch = 1
    return model_pool, skip_first_train, start_stage, start_epoch


def _train_stage(supernet,
                 teacher=None,
                 train_loader=None,
                 eval_loader=None,
                 optimizer=None,
                 scheduler=None,
                 loss_fn=None,
                 saver=None,
                 args=None,
                 stage=0,
                 model_pool=None,
                 start_epoch=1,
                 use_target=False,
                 writer=None):

    if args.reset_after_stage and stage > 0 and start_epoch == 1:
        if hasattr(supernet, 'module'):
            supernet.module.reset_params()
        else:
            supernet.reset_params()
    for epoch in range(start_epoch, args.step_epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if scheduler is not None:
            scheduler.step(epoch)
        loss = _train_epoch(supernet=supernet,
                            teacher=teacher,
                            train_loader=train_loader,
                            eval_loader=eval_loader,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss_fn,
                            args=args,
                            stage=stage,
                            epoch=epoch,
                            model_pool=model_pool,
                            use_target=use_target,
                            writer=writer,
                            saver=saver)
        if args.eval_intervals == 0 or epoch % args.eval_intervals == 0 or epoch == args.step_epochs:
            eval_loss = _evaluate(supernet=supernet,
                                  teacher=teacher,
                                  eval_data=eval_loader,
                                  loss_fn=loss_fn,
                                  args=args,
                                  stage=stage,
                                  epoch=epoch,
                                  use_target=use_target,
                                  writer=writer,
                                  model_pool=model_pool)
            if saver is not None:
                saver.save_checkpoint(supernet,
                                      optimizer,
                                      args,
                                      stage=stage,
                                      epoch=epoch,
                                      metric=eval_loss)

    return eval_loss


def _train_epoch(supernet,
                 teacher=None,
                 train_loader=None,
                 eval_loader=None,
                 optimizer=None,
                 scheduler=None,
                 loss_fn=None,
                 args=None,
                 stage=0,
                 epoch=0,
                 model_pool=None,
                 use_target=False,
                 writer=None,
                 saver=None):
    mse_weight = [0.0684, 0.171, 0.3422, 0.2395, 0.5474, 0.3422]

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    if args.print_detail:
        top1 = AverageMeter()
        ft_l_m = AverageMeter()
        sv_l_m = AverageMeter()
        klcos_m = AverageMeter()
        out_mean_m = AverageMeter()
        out_std_m = AverageMeter()
        a_d = AverageMeter()
        m_d = AverageMeter()
        klcos_loss_fn = KLCosineSimilarity()
    end = time.time()
    if args.label_train or use_target:
        ce_loss_fn = nn.CrossEntropyLoss().cuda()
    if args.feature_train:
        mse_loss_fn = nn.MSELoss().cuda()
    supernet.train()
    teacher.train()
    num_updates = epoch * len(train_loader)
    for step, (inputs, targets) in enumerate(train_loader):
        scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        # inputs = inputs.cuda()
        with torch.no_grad():
            guides, last_guide = teacher.forward(inputs, end_block=stage)
            if args.print_detail:
                guide_mean = guides.mean()
                guide_abs_mean = guides.abs().mean()
                guide_std = guides.std()
                guide_var = guides.var()
        # if use_target or args.label_train:
        #     targets = targets.cuda()
        if len(model_pool) > 0:
            partial_model = random.choice(model_pool)
        else:
            partial_model = None
        data_time_m.update(time.time() - end)
        # supernet.step_start_trigger()
        optimizer.zero_grad()
        for up_idx in range(args.update_frequency):
            if args.guide_input:
                logits, last_out, label_logits = supernet(last_guide,
                                                          encoding=partial_model,
                                                          start_block=stage - 1,
                                                          end_block=-1 if use_target else stage,
                                                          reverse_encod=args.reverse_train,
                                                          label_train=args.label_train)
            else:
                logits, last_out, label_logits = supernet(inputs,
                                                          encoding=partial_model,
                                                          end_block=-1 if use_target else stage,
                                                          reverse_encod=args.reverse_train,
                                                          return_last=args.save_last_feature,
                                                          label_train=args.label_train)
            # if not args.label_train:
            if use_target:
                loss = ce_loss_fn(logits, targets)
            else:
                assert (args.feature_train or args.label_train)
                feature_loss = 0
                supervise_loss = 0
                if args.feature_train:
                    feature_loss = mse_loss_fn(logits, guides)
                    loss = feature_loss
                if args.label_train:
                    supervise_loss = ce_loss_fn(label_logits, targets)
                    prec1 = accuracy(label_logits, targets)[0]
                    loss = supervise_loss
                if args.feature_train and args.label_train:
                    loss1 = args.loss_weight[0] * mse_weight[stage] * feature_loss
                    loss2 = args.loss_weight[1] * supervise_loss

                if args.print_detail:
                    with torch.no_grad():
                        # rkd_a_loss = rkd_a_loss_fn(logits, guides)
                        # rkd_d_loss = rkd_d_loss_fn(logits, guides)
                        klcos_loss = klcos_loss_fn(logits, guides)
                        out_mean = logits.mean()
                        out_std = logits.std()
                        abs_div = (logits - guides).abs().mean()
                        mean_dist = (out_mean - guide_mean).abs()

            torch.cuda.synchronize()
            if not args.distributed:
                losses_m.update(loss.item(), inputs.size(0))
                if args.print_detail and not use_target:
                    klcos_m.update(klcos_loss.item(), inputs.size(0))
                    out_mean_m.update(out_mean.item(), inputs.size(0))
                    out_std_m.update(out_std.item(), inputs.size(0))
                    a_d.update(abs_div.item(), inputs.size(0))
                    m_d.update(mean_dist.item(), inputs.size(0))
                    if args.label_train:
                        top1.update(prec1.item(), inputs.size(0))
                        sv_l_m.update(supervise_loss.item(), inputs.size(0))
            else:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), inputs.size(0))
                if args.print_detail and not use_target:
                    re_klc = reduce_tensor(klcos_loss.data, args.world_size)
                    re_o_m = reduce_tensor(out_mean.data, args.world_size)
                    re_o_s = reduce_tensor(out_std.data, args.world_size)
                    re_a_d = reduce_tensor(abs_div.data, args.world_size)
                    re_m_d = reduce_tensor(mean_dist.data, args.world_size)

                    klcos_m.update(re_klc.item(), inputs.size(0))
                    out_mean_m.update(re_o_m.item(), inputs.size(0))
                    out_std_m.update(re_o_s.item(), inputs.size(0))
                    a_d.update(re_a_d.item(), inputs.size(0))
                    m_d.update(re_m_d.item(), inputs.size(0))
                    if args.label_train:
                        re_top1 = reduce_tensor(prec1.data, args.world_size)
                        re_sv = reduce_tensor(supervise_loss.data, args.world_size)
                        top1.update(re_top1.item(), inputs.size(0))
                        sv_l_m.update(re_sv.item(), inputs.size(0))
                    if args.feature_train:
                        re_ft = reduce_tensor(feature_loss.data, args.world_size)
                        ft_l_m.update(re_ft.item(), args.world_size)
            # if mdl_idx == len(model_pool)-1 and up_idx == args.update_frequency-1:
            if args.label_train:
                loss2.backward()  # retain_graph=True)
                # loss1.backward()
            else:
                loss.backward()  # accumulate grad
            # else:
            #     loss.backward(retain_graph=True)

        optimizer.step()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        if writer is not None:
            step_num = (epoch - 1) * len(train_loader) + step
            writer.add_scalar('Stage{}/Loss'.format(stage), losses_m.val, step_num)
            if args.print_detail:
                grloss_val = a_d.val / (guide_abs_mean + 1e-9)
                writer.add_scalar('Stage{}/GRLoss'.format(stage),
                                  grloss_val,
                                  step_num)
                writer.add_scalar('Stage{}/KLCosLoss'.format(stage),
                                  klcos_m.val,
                                  step_num)
                if args.label_train:
                    writer.add_scalar('Stage{}/CELoss'.format(stage),
                                      supervise_loss.item(),
                                      step_num)
            if step % (4 * args.log_interval) == 0 and not use_target:
                input_img = inputs[0]
                guides_img = torchvision.utils.make_grid(guides[0].unsqueeze(1),
                                                         normalize=True,
                                                         scale_each=True)
                output_img = torchvision.utils.make_grid(logits[0].unsqueeze(1),
                                                         normalize=True,
                                                         scale_each=True)
                writer.add_image('Stage{}/Input'.format(stage), input_img, step_num)
                writer.add_image('Stage{}/Guide'.format(stage), guides_img, step_num)
                writer.add_image('Stage{}/Output'.format(stage), output_img, step_num)
                if stage > 0 and last_out is not None:
                    last_guide_img = torchvision.utils.make_grid(last_guide[0].unsqueeze(1),
                                                                 normalize=True,
                                                                 scale_each=True)
                    last_out_img = torchvision.utils.make_grid(last_out[0].unsqueeze(1),
                                                               normalize=True,
                                                               scale_each=True)
                    writer.add_image('Stage{}/LastGuide'.format(stage),
                                     last_guide_img,
                                     step_num)
                    writer.add_image('Stage{}/LastOut'.format(stage),
                                     last_out_img,
                                     step_num)

        if args.local_rank == 0 and step % args.log_interval == 0:
            if args.print_detail and not use_target:
                logging.info(
                    '\nTrain: stage {}, epoch {}, step [{:>4d}/{}]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.avg:.3f}s, {rate_avg:>7.2f}/s  '
                    'LR: {lr:.3e}  Data & Guide Time: {data_time.avg:.3f}\n'
                    'GuideMean: {guide_m:>.5f}  '
                    'GuideStd: {guide_d:>.5f}  '
                    'OutMean: {out_mean_m.val:>.5f} ({out_mean_m.avg:>.5f})  '
                    'OutStd: {out_std_m.val:>.5f} ({out_std_m.avg:>.5f})  '
                    'Dist_Mean: {m_d.val:>.5f} ({m_d.avg:>.5f})  \n'
                    'GRLoss: {dm_gm_val:>.5f} ({dm_gm:>.5f})  '
                    'CLLoss: {clloss_val:>.5f} ({clloss:>.5f})  '
                    'KLCosLoss: {klcos_m.val:>.5f} ({klcos_m.avg:>.5f})  \n'
                    'FeatureLoss: {ftlm.val:>.5f} ({ftlm.avg:>.5f})  '
                    'Top1Acc: {top1.val:>.5f}({top1.avg:>.5f}) \n'
                    'Relative MSE loss: {rela_loss_val:>.5f}({rela_loss_avg:>.5f})'
                    '\n'.format(
                        stage, epoch,
                        step, len(train_loader),
                        loss=losses_m,
                        guide_m=guide_mean,
                        guide_d=guide_std,
                        out_mean_m=out_mean_m,
                        out_std_m=out_std_m,
                        clloss=a_d.avg / (guide_std + 1e-9),
                        klcos_m=klcos_m,
                        dm_gm=a_d.avg / (guide_abs_mean + 1e-9),
                        clloss_val=a_d.val / (guide_std + 1e-9),
                        dm_gm_val=a_d.val / (guide_abs_mean + 1e-9),
                        m_d=m_d,
                        batch_time=batch_time_m,
                        rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                        svlm=sv_l_m,
                        ftlm=ft_l_m,
                        top1=top1,
                        rela_loss_avg=losses_m.avg / guide_var,
                        rela_loss_val=losses_m.val / guide_var
                    ))
            else:
                logging.info(
                    'Train: stage {}, epoch {}, step [{:>4d}/{}]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.avg:.3f}s, {rate_avg:>7.2f}/s  '
                    'LR: {lr:.3e}  '
                    'Data & Guide Time: {data_time.avg:.3f}'.format(
                        stage, epoch,
                        step, len(train_loader),
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))
        end = time.time()

        if saver is not None and args.eval_intervals == 0 and step % 500 == 0:
            # eval_loss = _evaluate(supernet=supernet,
            #                       teacher=teacher,
            #                       eval_data=eval_loader,
            #                       loss_fn=loss_fn,
            #                       args=args,
            #                       stage=stage,
            #                       epoch=epoch,
            #                       use_target=use_target,
            #                       writer=writer,
            #                       model_pool=model_pool)
            if saver is not None:
                saver.save_checkpoint(supernet,
                                      optimizer,
                                      args,
                                      stage=stage,
                                      epoch=epoch,
                                      step=step,
                                      metric=losses_m.avg)
        # one step end
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    return losses_m.avg


def _evaluate(supernet,
              teacher=None,
              eval_data=None,
              loss_fn=None,
              args=None,
              stage=0,
              epoch=0,
              use_target=False,
              writer=None,
              model_pool=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()

    supernet.eval()
    teacher.eval()

    end = time.time()
    # model_encoding = model_pool[random.randint(0, len(model_pool)-1)]
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(eval_data):
            if len(model_pool) > 0:
                partial_model = random.choice(model_pool)
            else:
                partial_model = None
            inputs = inputs.cuda()
            if use_target:
                targets = targets.cuda()
            guides, last_guide = teacher.forward(inputs, end_block=stage)
            if args.guide_input:
                logits, _, logits_supervise = supernet(last_guide,
                                                       start_block=stage - 1,
                                                       end_block=-1 if use_target else stage,
                                                       encoding=partial_model,
                                                       reverse_encod=args.reverse_train,
                                                       label_train=args.label_train)
            else:
                logits, _, logits_supervise = supernet(inputs,
                                                       end_block=-1 if use_target else stage,
                                                       encoding=partial_model,
                                                       reverse_encod=args.reverse_train,
                                                       label_train=args.label_train)
            if use_target:
                loss = loss_fn(logits, targets)
            else:
                loss = loss_fn(logits, guides)
                if args.label_train:
                    prec1 = accuracy(logits_supervise, targets)[0]
            torch.cuda.synchronize()
            if not args.distributed:
                losses_m.update(loss.item(), inputs.size(0))
                if args.label_train:
                    prec1_m.update(prec1.item(), inputs.size(0))
            else:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), inputs.size(0))
                if args.label_train:
                    reduced_prec1 = reduce_tensor(prec1.data, args.world_size)
                    prec1_m.update(reduced_prec1.item(), inputs.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

        if args.local_rank == 0:
            logging.info(
                'Random Test: stage {}, epoch {}  '
                'Loss: {loss.avg:>6.4f}  '
                'Prec@1: {prec1.avg:>.4f}  '
                'Time: {batch_time.avg:.3f}s, {rate_avg:>7.2f}/s  '.format(
                    stage, epoch,
                    loss=losses_m,
                    prec1=prec1_m,
                    batch_time=batch_time_m,
                    rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg))

    # if args.label_train:
    #     return prec1_m.avg
    # else:
    return losses_m.avg


def _potential(supernet,
               teacher=None,
               eval_loader=None,
               loss_fn=None,
               args=None,
               stage=None,
               stage_model_pool=None,
               use_target=False,
               reset_data=None):
    r"""
    Evaluate the potential of one model.
    """
    batch_time_m = AverageMeter()
    loss_l = []
    prec1_l = []
    sq_m_l = []
    a_d_l = []
    rl_l = []
    gr_l = []
    cl_l = []
    if args.print_detail:
        top1 = AverageMeter()
        klcos_m = AverageMeter()

        m_d = AverageMeter()

    # supernet.eval()
    for layer in supernet.module.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or (
                has_apex and isinstance(layer, apex.parallel.SyncBatchNorm)):
            print(layer, '\n supernet.bn.train()')
            layer.train()
    # teacher.eval()
    for layer in teacher.module.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or (
                has_apex and isinstance(layer, apex.parallel.SyncBatchNorm)):
            print(layer, '\n teacher.bn.train()')
            layer.train()

    for step, (inputs, targets) in enumerate(eval_loader):
        if step == args.potential_eval_times - 1:
            break
        end = time.time()
        with torch.no_grad():
            if not args.prefetcher:
                inputs = inputs.cuda()
                if use_target:
                    targets = targets.cuda()
            if args.reset_bn_eval:
                reset_inputs, _ = reset_data
                bn_resetter = BN_Correction(teacher, reset_inputs, args)
                (reset_guides, reset_last_guide) = bn_resetter(end_block=stage,
                                                               net='teacher')
                # reset_guides, reset_last_guide = teacher.forward(reset_inputs, end_block=stage)
            guides, last_guide = teacher.forward(inputs, end_block=stage)
            if args.print_detail:
                guide_mean = guides.mean()
                guide_abs_mean = guides.abs().mean()
                guide_std = guides.std()
                guide_var = guides.var()

        for mdl_idx, model_encoding in enumerate(stage_model_pool):

            if step == 0:
                losses_m = AverageMeter()
                prec1_m = AverageMeter()
                sq_m = AverageMeter()
                a_d = AverageMeter()
                gr = AverageMeter()
                cl = AverageMeter()
                rl = AverageMeter()
                loss_l.append(losses_m)
                prec1_l.append(prec1_m)
                a_d_l.append(a_d)
                sq_m_l.append(sq_m)
                rl_l.append(rl)
                gr_l.append(gr)
                cl_l.append(cl)
            with torch.no_grad():
                if args.guide_input:
                    if args.reset_bn_eval:
                        bn_resetter = BN_Correction(supernet, reset_last_guide, args)
                        bn_resetter(encoding=model_encoding,
                                    start_block=stage - 1,
                                    end_block=-1 if use_target else stage,
                                    reverse_encod=args.reverse_train,
                                    net='supernet')
                    logits, _, logits_supervise = supernet(last_guide,
                                                           encoding=model_encoding,
                                                           start_block=stage - 1,
                                                           end_block=-1 if use_target else stage,
                                                           reverse_encod=args.reverse_train)
                else:
                    if args.reset_bn_eval:
                        bn_resetter = BN_Correction(supernet, reset_last_guide, args)
                        bn_resetter(encoding=model_encoding,
                                    start_block=-1,
                                    end_block=-1 if use_target else stage,
                                    reverse_encod=args.reverse_train,
                                    net='supernet')
                    logits, _, logits_supervise = supernet(inputs,
                                                           encoding=model_encoding,
                                                           end_block=-1 if use_target else stage,
                                                           reverse_encod=args.reverse_train)
                if use_target:
                    loss = loss_fn(logits, targets)
                    prec1 = accuracy(logits, targets)[0]
                else:
                    loss = loss_fn(logits, guides)
                    # if args.label_train:
                    #     prec1 = accuracy(logits_supervise, targets)[0]
                    if args.print_detail:
                        with torch.no_grad():
                            sqrt_rela_loss = math.sqrt(loss) / guide_var
                            abs_div = (logits - guides).abs().mean()
                            rl = loss / (guide_var + 1e-9)
                            gr = abs_div / (guide_abs_mean + 1e-9)
                            cl = abs_div / (guide_std + 1e-9)
                torch.cuda.synchronize()
                if not args.distributed:
                    loss_l[mdl_idx].update(loss.item(), inputs.size(0))
                    if use_target:
                        prec1_l[mdl_idx].update(prec1.item(), inputs.size(0))
                    else:
                        sq_m_l[mdl_idx].update(sqrt_rela_loss.item(), inputs.size(0))
                        a_d_l[mdl_idx].update(abs_div.item(), inputs.size(0))
                else:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    loss_l[mdl_idx].update(reduced_loss.item(), inputs.size(0))
                    if use_target:
                        re_prec1 = reduce_tensor(prec1.data, args.world_size)
                        prec1_l[mdl_idx].update(re_prec1.item(), inputs.size(0))
                    else:
                        re_sq = reduce_tensor(sqrt_rela_loss.data, args.world_size)
                        sq_m_l[mdl_idx].update(re_sq.item(), inputs.size(0))
                        re_a_d = reduce_tensor(abs_div.data, args.world_size)
                        a_d_l[mdl_idx].update(re_a_d.item(), inputs.size(0))
                        re_rl = reduce_tensor(rl.data, args.world_size)
                        re_gr = reduce_tensor(gr.data, args.world_size)
                        re_cl = reduce_tensor(cl.data, args.world_size)
                        rl_l[mdl_idx].update(re_rl.item(), inputs.size(0))
                        gr_l[mdl_idx].update(re_gr.item(), inputs.size(0))
                        cl_l[mdl_idx].update(re_cl.item(), inputs.size(0))
                    # if args.label_train:
                    #     reduced_prec1 = reduce_tensor(prec1.data, args.world_size)
                    #     prec1_m.update(reduced_prec1.item(), inputs.size(0))

                batch_time_m.update(time.time() - end)
                # print('model {}({}): {} '.format(mdl_idx, args.local_rank, loss))

            if args.local_rank == 0:  # and (step % 5 == 0 or step == args.potential_eval_times - 1):
                logging.info(
                    'validate model {}({}/{}):  '
                    'Loss: {loss.avg:>6.4f}  '
                    'Acc: {prec1.avg:>6.4f}  '
                    'MSE/var: {rela_loss_avg:>6.4f}  '
                    'sqrt(MSE)/std: {sqrt_rela_loss:>6.4f}  '
                    'GRLoss: {grloss:>6.4f}  '
                    'CLLoss: {clloss:>6.4f}  '
                    'Time: {batch_time.avg:.3f}s, {rate_avg:>7.2f}/s  '.format(
                        mdl_idx,
                        step,
                        args.potential_eval_times - 1 if args.potential_eval_times != 0 else len(
                            eval_loader) - 1,
                        loss=loss_l[mdl_idx],
                        rela_loss_avg=rl_l[mdl_idx].avg,
                        sqrt_rela_loss=sq_m_l[mdl_idx].avg,
                        grloss=gr_l[mdl_idx].avg,
                        clloss=cl_l[mdl_idx].avg,
                        prec1=prec1_l[mdl_idx],
                        batch_time=batch_time_m,
                        rate=inputs.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg))

    # if args.label_train:
    #     return prec1_m.avg
    # else:
    for idx, loss in enumerate(loss_l):
        loss_l[idx] = loss.avg
        sq_m_l[idx] = sq_m_l[idx].avg
        gr_l[idx] = gr_l[idx].avg
        cl_l[idx] = cl_l[idx].avg
        rl_l[idx] = rl_l[idx].avg
        prec1_l[idx] = prec1_l[idx].avg

    potent = {}
    potent['loss'] = loss_l
    potent['rela_loss'] = rl_l
    potent['sq_loss'] = sq_m_l
    potent['gr_loss'] = gr_l
    potent['cl_loss'] = cl_l
    potent['prec1'] = prec1_l
    if args.local_rank == 0:
        print(potent)
    return potent


def get_all_models(supernet,
                   stage,
                   model_pool,
                   args):
    r"""
    Get all possible model encodings.
    And concat with models in model_pool if necessary.
    """
    if hasattr(supernet, 'module'):
        supernet = supernet.module
    num_layers = supernet.get_layers(stage)
    num_of_ops = supernet._num_of_ops

    idx = list(range(num_of_ops))
    stage_model_pool = list(itertools.product(idx, repeat=num_layers))
    for i, model in enumerate(stage_model_pool):
        stage_model_pool[i] = list(stage_model_pool[i])
    if not args.guide_input and len(model_pool) > 0:
        if not args.reverse_train:
            stage_model_pool = list(itertools.product(model_pool, stage_model_pool))
        else:
            stage_model_pool = list(itertools.product(stage_model_pool, model_pool))
        for mdl_idx, model in enumerate(stage_model_pool):
            stage_model_pool[mdl_idx] = list(model[0]) + list(model[1])
    return list(stage_model_pool)


def get_best_items(candidates=None,
                   value=None,
                   top_num=5,
                   best='top'):
    r"""
    modified from _get_top_items()
    """
    if best == 'top':
        top_index = np.argsort(-np.array(value))
    else:
        top_index = np.argsort(np.array(value))
    top_items = []
    top_value = []
    top_num = min(top_num, len(candidates))
    for i in range(top_num):
        top_items.append(candidates[top_index[i]].copy())
        top_value.append(value[top_index[i]])
    return top_items, top_value, top_index.tolist()


if __name__ == '__main__':

    model_pool = [[2, 1], [2, 0]]
    layers = 3
    numop = 4
    idx = list(range(numop))
    all_net_idx = list(itertools.product(idx, repeat=layers))

    stage_model_pool = list(itertools.product(model_pool, all_net_idx))
    for i, model in enumerate(stage_model_pool):
        stage_model_pool[i] = model[0] + list(model[1])

    print(stage_model_pool)
