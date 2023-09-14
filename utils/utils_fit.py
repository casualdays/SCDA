import os

import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm
from tqdm.contrib import tzip
from itertools import cycle
from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, s_gen, t_gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    # train_loader = tzip(s_gen, t_gen)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, position=0, mininterval=0.3)
    model_train.train()


    for iteration, (s_batch, t_batch) in enumerate(tzip(s_gen, t_gen)):
        if iteration >= epoch_step:
            break
        s_imgs, s_pngs, s_labels = s_batch
        t_imgs, t_pngs, t_labels = t_batch


        with torch.no_grad():
            weights = torch.Tensor(cls_weights)
            if cuda:
                s_imgs    = s_imgs.cuda(local_rank)
                s_pngs    = s_pngs.cuda(local_rank)
                s_labels  = s_labels.cuda(local_rank)

                t_imgs = t_imgs.cuda(local_rank)
                t_pngs = t_pngs.cuda(local_rank)
                t_labels = t_labels.cuda(local_rank)

                weights = weights.cuda(local_rank)
        #----------------------#
        # Zero out the gradients.
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #
            #----------------------#
            lossfn, outputs = model_train(s_imgs, t_imgs, s_labels, s_pngs, weights)
            # outputs = model_train()
            #----------------------#
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, s_pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, s_pngs, weights, num_classes = num_classes)

            if dice_loss:
                # main_dice = Dice_loss(outputs, labels)
                # loss = None
                # for index in range(len(lossfn)):
                #     loss = loss + lossfn[index]
                # loss = loss + lossfn[0] + lossfn[1] + 0.1*lossfn[2] + lossfn[3] + lossfn[4] + 10*lossfn[5] +lossfn[6]
                loss = loss + lossfn[0] + lossfn[1] + 0.1 * lossfn[2] + lossfn[3] + lossfn[4]
                # loss = loss + lossfn[0] + lossfn[1]
                # loss = loss + lossfn[0] + 2.5*lossfn[1] + lossfn[2]
                # loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                # -------------------------------#
                _f_score = f_score(outputs, s_labels)


            #----------------------#
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #----------------------#
                outputs = model_train(s_imgs)
                #----------------------#
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, s_pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, s_pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, s_labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #-------------------------------#
                    _f_score = f_score(outputs, s_labels)
                    
            #----------------------#
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)


    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #----------------------#
            outputs = model_train(imgs, imgs, labels)
            #----------------------#
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            #-------------------------------#
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))