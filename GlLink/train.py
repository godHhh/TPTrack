import os
import torch
import time
from torch.optim import Adam
from datetime import datetime
from os.path import join, exists
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import GlLink.config as cfg
from GlLink.model import PostLinker
from GlLink.dataset import LinkData

from loguru import logger
from yolox.utils import setup_logger

from GlLink.OurLoss import OurLoss


def train(save: bool):
    setup_logger(cfg.model_savedir, filename="val_log.txt", mode="a")
    logger.info("开始训练")
    model = PostLinker()
    model.cuda()
    model.train()
    dataset = LinkData(cfg.root_train, 'train')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True
    )
    loss_fn = OurLoss(softmax=True, epsilon=-0.6)
    name = "Loss_ablation"
    optimizer = Adam(model.parameters(), lr=cfg.train_lr, weight_decay=cfg.train_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train_epoch, eta_min=1e-5)

    now_date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    min_loss = 100.0
    best_model_epoch = 0
    loss_list = []
    print('======================= Start Training =======================')
    for epoch in range(cfg.train_epoch):
        logger.info('epoch: %d with lr=%.0e' % (epoch, optimizer.param_groups[0]['lr']))
        loss_sum = 0
        for i, (pair1, pair2, pair3, pair4, label) in enumerate(dataloader):
            optimizer.zero_grad()
            pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).cuda()
            pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
            pairs_3 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
            pairs_4 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
            label = torch.cat(label, dim=0).cuda()
            output = model(pairs_1, pairs_2, pairs_3, pairs_4)
            loss = loss_fn(output, label)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.info('loss: {:.3f}'.format(loss_sum))
        loss_list.append(loss_sum)
        if loss_sum < min_loss:
            best_model_epoch = epoch
            torch.save(model.state_dict(), join(cfg.model_savedir, 'best_epoch_{}_{}.pth'.format(now_date, name)))
            min_loss = loss_sum
    logger.info('第{}轮epoch是最好的模型'.format(best_model_epoch))
    with open("{}_loss_{}_{}.txt".format('Our', now_date, name), 'w') as file:
        for line in loss_list:
            file.write(str(line) + '\n')
    file.close()
    if save:
        if not exists(cfg.model_savedir):
            os.mkdir(cfg.model_savedir)
        torch.save(model.state_dict(), join(cfg.model_savedir, 'epoch{}_{}_{}.pth'.format(epoch + 1, now_date, name)))
    return model


def validate(model):
    model.eval()
    dataset = LinkData(cfg.root_train, 'val')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.val_batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False
    )
    labels = list()
    outputs = list()
    for i, (pair1, pair2, pair3, pair4, label) in enumerate(dataloader):
        pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).cuda()
        pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
        label = torch.cat(label, dim=0).cuda()
        output = model(pairs_1, pairs_2)
        labels.extend(label.tolist())
        outputs.extend(output.tolist())
    outputs = [0 if x[0] > x[1] else 1 for x in outputs]
    precision = precision_score(labels, outputs, average='macro', zero_division=0)
    recall = recall_score(labels, outputs, average='macro', zero_division=0)
    f1 = f1_score(labels, outputs, average='macro', zero_division=0)
    confusion = confusion_matrix(labels, outputs)
    print('  f1/p/r: {:.2f}/{:.2f}/{:.2f}'.format(f1, precision, recall))
    print('  ConfMat: ', confusion.tolist())
    model.train()


if __name__ == '__main__':
    print(datetime.now())
    train(save=True)
    print(datetime.now())
    logger.info("end")
