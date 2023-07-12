# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from torch.optim import AdamW


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    # for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        time_start = time.time()
        for i, (trains, labels) in enumerate(train_iter):
            # print('i=', i)
            outputs = model(trains)
            model.zero_grad()
            # print('outputs=', outputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch finished...')
        train_acc, tran_loss = evaluate(config, model, train_iter)
        dev_acc, dev_loss = evaluate(config, model, dev_iter)
        msg = 'Train Loss: {0:>3.4}, Train Acc: {1:>6.4%}'
        print(msg.format(tran_loss, train_acc))
        msg = 'Val Loss: {0:>3.4}, Val Acc: {1:>6.4%}'
        print(msg.format(dev_loss, dev_acc))

        if (epoch + 1) % config.save_epoch == 0:
            torch.save(model.state_dict(), config.save_path + 'model_' + str(epoch + 1) + 'e.ckpt')

        time_end = time.time()
        time_sum = time_end - time_start
        print('Finish current epoch! Using %.2fs' % time_sum)



def test(config, model, test_iter, witch_epoch,  print_all=True):
    print('Test Report for ' + witch_epoch)
    model.load_state_dict(torch.load(config.save_path + witch_epoch))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.4},  Test Acc: {1:>6.4%}'
    print(msg.format(test_loss, test_acc))
    if print_all == True:
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
