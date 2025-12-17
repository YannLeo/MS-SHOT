from pathlib import Path

import numpy as np
import torch
import tqdm
from scipy.spatial.distance import cdist
from torch import nn

import datasets
import models
import torch.nn.functional as F

from .basic_trainer import BasicTrainer


def Entropy(input_):
    bs = input_.size(0) 
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class SHOTNormEMATrainer(BasicTrainer):
    def __init__(self, info: dict, resume=None, path=Path(), Rx_s=None, Rx_t=None, device=torch.device('cuda')):
        if 'resume' in info:
            import os
            path1 = Path(info['save_dir']) / (Rx_s + '_' + Rx_t) / (Rx_s + '_' + Rx_t + '_sourceonly') # saved
            dirs = os.listdir(path1)
            dirs.sort()
            # resume = path1 / dirs[-1] / 'model' / 'checkpoint-epoch18.pth'
            resume = path1 / dirs[-1] / 'model' / 'model_best.pth'
        # info['epochs'] = 20
        self.uneven = info['uneven'] if 'uneven' in info else False
        if self.uneven:
            self.uneven_num = info['uneven_num'] if 'uneven_num' in info else None
            self.uneven_ratio = info['uneven_ratio'] if 'uneven_ratio' in info else None
        super().__init__(info, resume, path, Rx_t, Rx_t, device)
        # self.loss_func = nn.CrossEntropyLoss()
        self.epoch = 0
        args_target_eval = info['dataloader_train']
        args_target_eval['args']['shuffle'] = False
        args_target_eval['args']['drop_last'] = False
        args_target_eval['dataset']['args']['Rx'] = [Rx_t]
        # self.dataset_target_eval = self._get_object(datasets, info['dataloader_train']['dataset']['name'],
        #                                        info['dataloader_train']['dataset']['args'])
        self.dataloader_target_eval = torch.utils.data.DataLoader(dataset=self.dataset_train, **args_target_eval['args'])
        self.lock(self.model.netC)
        if 'classifier_weight' in info:
            self.classifier_weight = info['classifier_weight']
        if 'fbnm_weight' in info:
            self.fbnm_weight = info['fbnm_weight']
        if 'nmlzabs_weight' in info:
            self.nmlzabs_weight = info['nmlzabs_weight']
        if 'temperature' in info:
            self.temperature = info['temperature']
        else:
            self.temperature = 1
        self.initc = None
        if 'ema_initc_par' in info:
            self.ema_initc_par = info['ema_initc_par']
            # self.ema_initc_par = 0.995
        

    def _prepare_opt(self, info):
        param_group = []
        lr = 0.01
        for k, v in self.model.netF.named_parameters():
            param_group += [{'params': v, 'lr': lr*0.1}]
        for k, v in self.model.netB.named_parameters():
            param_group += [{'params': v, 'lr': lr}]
        for k, v in self.model.netC.named_parameters():
            param_group += [{'params': v, 'lr': lr}]   
        self.opt = torch.optim.SGD(params=param_group)
        # self.opt = torch.optim.Adam(params=param_group)
        # self.opt = torch.optim.Adam(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                              {'optimizer': self.opt, **info['lr_scheduler']['args']})  

    def lock(self, model):
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False

    def sharpen(self, x, T):
        return x**(1/T) / np.sum(x**(1/T), axis=1).reshape(x.shape[0], 1)

    def model_train(self):
        self.model.netB.train()
        self.model.netF.train()

    def model_eval(self):
        self.model.netB.eval()
        self.model.netF.eval()

    def obtain_label(self):
        self.model_eval()
        loader = self.dataloader_target_eval
        netF = self.model.netF
        netB = self.model.netB
        netC = self.model.netC

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.__next__()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feas = netB(netF(inputs))
                outputs = netC(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()

        # if True:
        if 'ema_initc' in self.info and self.info['ema_initc']:
            if self.initc is not None:
                initc1 = aff.transpose().dot(all_fea)
                initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])
                initc = 0.95 * self.initc + 0.05 * initc1
            else:
                initc = aff.transpose().dot(all_fea)
                initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        else:
            # if 'emac' in self.info and self.info['emac'] and self.initc is not None:
            #     initc = self.initc
            # else:
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        # if self.initc is None:
        # initc = aff.transpose().dot(all_fea)
        # initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        # else:
            # initc1 = aff.transpose().dot(all_fea)
            # initc1 = initc / (1e-8 + aff.sum(axis=0)[:,None])
            # initc = 0.9 * self.initc + 0.1 * initc1

        dd = cdist(all_fea, initc, 'cosine')
        if 'soft_label' in self.info and self.info['soft_label']:
            dd = torch.from_numpy(dd)
            pred_label = nn.Softmax(dim=1)(-1 * dd).numpy()
            pred_label = self.sharpen(pred_label, self.temperature)
        else:
            pred_label = dd.argmin(axis=1)
            pred_label = np.eye(K)[pred_label]
        acc = np.sum(pred_label.argmax(axis=1) == all_label.float().numpy()) / len(all_fea)

        for _ in range(1):
            # if 'soft_label' in self.info and self.info['soft_label']:
            #     aff = pred_label
            # else:
            #     aff = np.eye(K)[pred_label]
            aff = pred_label
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea, initc, 'cosine')
            if 'soft_label' in self.info and self.info['soft_label']:
                dd = torch.from_numpy(dd)
                pred_label = nn.Softmax(dim=1)(-1 * dd).numpy()
                pred_label = self.sharpen(pred_label, self.temperature)
            else:
                pred_label = dd.argmin(axis=1)
                pred_label = np.eye(K)[pred_label]
            acc = np.sum(pred_label.argmax(axis=1) == all_label.float().numpy()) / len(all_fea)

        self.model_train()
        self.pred_label = torch.from_numpy(pred_label)
        # self.distance = torch.softmax(torch.from_numpy(-dd).float() / self.temperature, dim=1)
        # self.distance = self.pred_label
        self.initc = initc

        return accuracy, acc

    def get_pseudo_label(self, feat):
        feat = torch.cat((feat, torch.ones(feat.size(0), 1)), 1)
        feat = (feat.t() / torch.norm(feat, p=2, dim=1)).t()
        feat = feat.float().cpu().numpy()
        dd = cdist(feat, self.initc, 'cosine')
        dd = torch.from_numpy(dd)
        label = nn.Softmax(dim=1)(-1 * dd).numpy()
        label = self.sharpen(label, self.temperature)
        return label, feat
    
    def update_c(self, label, feat):
        delta = label.transpose().dot(feat)
        self.initc = self.initc * self.ema_initc_par + delta * (1 - self.ema_initc_par)


    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        The main training process
        """
        # helper variables
        num_samples, num_correct = 0, 0  
        train_loss = 0. 

        self.model_train()  # don't forget
        loop = tqdm.tqdm(enumerate(self.dataloader_train), total=self.num_batch_train, leave=False, 
                         desc=f"Epoch {epoch}/{self.max_epoch}")
        for batch, (data, targets, index) in loop:
            if ('interval_iter' in self.info and batch % self.info['interval_iter'] == 0) or batch == 0:
                accuracy, acc = self.obtain_label()

            data, targets = data.to(self.device), targets.to(self.device)
            # pseudo_targets = self.pred_label[index].to(self.device)

            # 1. forwarding
            predicts, feature = self.model(data, return_feature=True)

            # 2. computing loss
            if 'soft_label' in self.info and self.info['soft_label']:
                label, feat = self.get_pseudo_label(feature.detach().cpu())
                soft_pseudo_targets = torch.from_numpy(label).to(self.device)
                pseudo_targets = torch.argmax(soft_pseudo_targets, dim=1)
                # 根据熵来改变温度
                # soft_pseudo_targets_entropy = torch.mean(Entropy(soft_pseudo_targets))
                # if soft_pseudo_targets_entropy < 0.6:
                #     self.temperature *= 0.8
                #     self.temperature = 0.01 if self.temperature < 0.01 else self.temperature
                # else:
                #     self.temperature *= 1.2
                #     self.temperature = 0.5 if self.temperature > 0.5 else self.temperature
                # 根据熵来改变classifier_weight
                # soft_pseudo_targets_entropy = torch.mean(Entropy(soft_pseudo_targets))
                # if soft_pseudo_targets_entropy < 0.6:
                #     self.classifier_weight *= 1.2
                #     self.classifier_weight = 0.5 if self.classifier_weight > 0.5 else self.classifier_weight
                # else:
                #     self.classifier_weight *= 0.8
                #     self.classifier_weight = 0.05 if self.classifier_weight < 0.05 else self.classifier_weight
                # 根据熵来选择软标签或硬标签 
                # pseudo_targets_onehot = F.one_hot(soft_pseudo_targets.argmax(dim=1), num_classes=6).double()
                # soft_pseudo_targets_entropy = Entropy(soft_pseudo_targets).cpu().numpy()
                # index_confidence = np.where(soft_pseudo_targets_entropy < 0.6)
                # soft_pseudo_targets[index_confidence,:] = pseudo_targets_onehot[index_confidence,:]

                # 计算 self.initc 中每行样本之间的平均距离
                dist = torch.cdist(torch.from_numpy(self.initc), torch.from_numpy(self.initc), p=2)

                # print(dist.mean())
                if 'emac' in self.info and self.info['emac']:
                    self.update_c(label, feat)
            else:
                label, feat = self.get_pseudo_label(feature.detach().cpu())
                pseudo_targets = torch.argmax(torch.from_numpy(label), dim=1).to(self.device)
                if 'emac' in self.info and self.info['emac']:
                    self.update_c(label, feat)

            if 'soft_label' in self.info and self.info['soft_label']:
                classifier_loss = self.loss_func(predicts, soft_pseudo_targets) * self.classifier_weight
            elif 'none_label' in self.info and self.info['none_label']:
                classifier_loss = 0
            else:
                classifier_loss = self.loss_func(predicts, pseudo_targets) * self.classifier_weight

            softmax_out = nn.Softmax(dim=1)(predicts)
            msoftmax = softmax_out.mean(dim=0)
            entropy_loss = torch.tensor(0.).to(self.device)
            im_loss = torch.tensor(0.).to(self.device)

            if 'fbnm' in self.info and self.info['fbnm']:
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_out,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_out.shape[0],softmax_out.shape[1])])
                entropy_loss += fbnm_loss * self.fbnm_weight
            if 'rowsparse' in self.info and self.info['rowsparse']:
                rowsparse_loss = - torch.mean(torch.linalg.norm(softmax_out, ord=2, dim=1)) * 2.0
                entropy_loss += rowsparse_loss
            if 'entropy' in self.info and self.info['entropy']:
                entropy_loss += torch.mean(Entropy(softmax_out))
            im_loss += entropy_loss

            if 'onm' in self.info and self.info['onm']:
                im_loss += msoftmax.max() * self.num_classes * 0.4
            if 'nmlzpow' in self.info and self.info['nmlzpow']:
                im_loss += (msoftmax * self.num_classes - 1).pow(2).sum() * 0.3
            if 'nmlzabs' in self.info and self.info['nmlzabs']:
                if self.uneven:
                    # print('uneven')
                    if self.uneven_num is not None:
                        distribute = torch.cat([0.5 * torch.ones(self.uneven_num), 1 * torch.ones(self.num_classes - self.uneven_num)]).cuda()
                    elif self.uneven_ratio is not None:
                        distribute = torch.tensor(self.uneven_ratio).cuda()
                    distribute = distribute / distribute.sum()
                    im_loss += (msoftmax * self.num_classes - distribute * self.num_classes).abs().sum() * self.nmlzabs_weight
                    # print(distribute * self.num_classes)
                else:
                    im_loss += (msoftmax * self.num_classes - 1).abs().sum() * self.nmlzabs_weight
                    # print(msoftmax * self.num_classes)
            if 'nearpow' in self.info and self.info['nearpow']:
                diffs = torch.unsqueeze(msoftmax, 0) - torch.unsqueeze(msoftmax, 1) * 0.3
                im_loss += (diffs * diffs).sum()/2
            if 'nearabs' in self.info and self.info['nearabs']:
                diffs = torch.unsqueeze(msoftmax, 0) - torch.unsqueeze(msoftmax, 1) * 0.6
                im_loss += diffs.abs().sum()/2
            if 'balance' in self.info and self.info['balance']:
                im_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            # loss = classifier_loss + im_loss + dist.mean() * 0.3
            loss = classifier_loss + im_loss

            # 3. backwarding: compute gradients and update parameters
            self._reset_grad()
            loss.backward()
            self.opt.step()
            
            # 4. updating learning rate by step; move it to self.train() if you want to update lr by epoch
            self.metrics_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], epoch*self.num_batch_train+batch)
            self.lr_scheduler.step()
            
            # 5. computing metrics
            num_samples += data.shape[0]
            train_loss += loss.item()
            num_correct += torch.sum(predicts.argmax(dim=1) == pseudo_targets).item()
            
            # display at the end of the progress bar
            # if batch % (__interval:=1 if self.num_batch_train > 10 else self.num_batch_train // 10) == 0:
            if batch % (__interval:=1) == 0:
                loop.set_postfix(loss_step=f"{loss.item():.3f}", refresh=False)

        return {
            "train_loss": train_loss / self.num_batch_train,
            "pseudo_acc_before_kmeans": accuracy,
            "pseudo_acc": acc,
            "train_acc": (num_correct / num_samples, 'blue'),  # (value, color) is supported,
            "classifier_loss": classifier_loss,
        }

    def test_epoch(self, epoch):
        self._y_pred, self._y_true = [], []  
        num_correct, num_samples = 0, 0
        test_loss = 0.
        
        self.model_eval()  # don't forget
        with torch.no_grad():
            for data, targets, index in self.dataloader_test:
                if self.plot_confusion:
                    self._y_true.append(targets.numpy())
                    
                data, targets = data.to(self.device), targets.to(self.device)
                num_samples += data.shape[0]
                # forwarding
                predicts = self.model(data)
                # computing metrics
                test_loss += self.loss_func(predicts, targets).item()
                pred_labels = predicts.argmax(dim=1)
                num_correct += torch.sum(pred_labels == targets).item()
                
                if self.plot_confusion:
                    self._y_pred.append(pred_labels.cpu().numpy())
                    
        return {
            "test_loss": test_loss / self.num_batch_test,
            "test_acc": (num_correct / num_samples, 'red'),  # (value, color) is supported
        }

    