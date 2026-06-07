import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import math
import pandas as pd
import torch.nn as nn
import random
from torch.nn import init
from sklearn.metrics import confusion_matrix

import utils
import graph
from load_data_seed import load_data


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)

cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if cuda else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log = []

# Command setting
parser = argparse.ArgumentParser(description='UF_AMA')
parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_epoch', type=int, default=400)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--lamb', type=float, default=0.5)
parser.add_argument('--trans_loss', type=str, default='mmd')
parser.add_argument('--gamma', type=int, default=1, help='the fc layer and the sharenet have different or same learning rate')
args = parser.parse_args(args=[])


def setup_seed(seed): ## set up the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weight_init(m):  ## model parameter initialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
        # torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()

def segmented_function(epoch):
    if epoch <= 10:
        value = 1
    elif 10 < epoch <= 40:
        value = 2 / (1 + math.exp(-10 * (args.n_epoch) / args.n_epoch)) - 1.5
    elif 40 < epoch <= 85:
        value = 1 * np.exp(-0.6 * epoch)
    else:
        value = 0

    return value

def evaluate(model, target_test_loader):
    model.eval()
    correct = 0

    len_target_dataset = len(target_test_loader.dataset)

    num_classes = args.n_class
    conf_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for eeg_target, eye_target, label_target in target_test_loader:
            eeg_target, eye_target, label_target = eeg_target.to(DEVICE), eye_target.to(DEVICE), label_target.to(DEVICE)

            output = model.predict(eeg_target, eye_target)
            prediction = torch.max(output, 1)[1]

            label_target_1D = torch.argmax(label_target, dim=1)
            correct += torch.sum(prediction == label_target_1D)
            conf_matrix += confusion_matrix(label_target_1D.cpu().numpy(), prediction.cpu().numpy(), labels=np.arange(num_classes))

            acc = 100. * correct / len_target_dataset

    return acc, prediction, conf_matrix

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, cls_weight, distill_weight, sub_id):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    best_acc = 0
    stop = 0
    best_confusion_matrix = None
    
    model_filename = f'./model_parameters/seed/model_parameters_subject_{sub_id}.pth'
    os.makedirs(f'./model_parameters/seed', exist_ok=True)
    
    # cls_loss_sum = 0
    for e in range(args.n_epoch):
        stop += 1

        train_loss_clf = utils.AverageMeter()
        train_loss_cls = utils.AverageMeter()
        train_loss_align_mmd = utils.AverageMeter()
        train_loss_align_single = utils.AverageMeter()
        train_loss_align_fusion = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()

        model.train()
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()

        for mlen in range(n_batch):
            eta = 0.001
            eeg_source, eye_source, label_source = next(iter_source)
            eeg_target, eye_target, label_target = next(iter_target)

            if mlen % len(target_train_loader) == 0:
                iter_target = iter(target_train_loader)
            if cuda:
                eeg_source, eye_source, label_source = eeg_source.cuda(), eye_source.cuda(), label_source.cuda()
                eeg_target, eye_target, label_target = eeg_target.cuda(), eye_target.cuda(), label_target.cuda()

            eeg_source, eye_source, label_source = eeg_source.to(DEVICE), eye_source.to(DEVICE), label_source.to(DEVICE)
            eeg_target, eye_target, label_target = eeg_target.to(DEVICE), eye_target.to(DEVICE), label_target.to(DEVICE)

            optimizer.zero_grad()
            (eeg_source_clf, eeg_mmd_loss, eeg_distill_loss, eeg_sim_matrix,
             eye_source_clf, eye_mmd_loss, eye_distill_loss, eye_sim_matrix,
             fusion_source_clf, fusion_mmd_loss, fusion_target_clf_loss, fusion_sim_matrix,
             estimated_sim_truth) = model(e, eeg_source, eye_source, eeg_target, eye_target, label_source)

            eeg_clf_loss = criterion(eeg_source_clf, label_source)
            eye_clf_loss = criterion(eye_source_clf, label_source)
            fusion_clf_loss = criterion(fusion_source_clf, label_source)

            eeg_bce_loss = (-(torch.log(eeg_sim_matrix + eta) * estimated_sim_truth)
                            - (1 - estimated_sim_truth) * torch.log(1 - eeg_sim_matrix + eta))
            eye_bce_loss = (-(torch.log(eye_sim_matrix + eta) * estimated_sim_truth)
                            - (1 - estimated_sim_truth) * torch.log(1 - eye_sim_matrix + eta))
            fusion_bce_loss = (-(torch.log(fusion_sim_matrix + eta) * estimated_sim_truth)
                               - (1 - estimated_sim_truth) * torch.log(1 - fusion_sim_matrix + eta))
            
            eeg_cls_loss = torch.mean(eeg_bce_loss)
            eye_cls_loss = torch.mean(eye_bce_loss)
            fusion_cls_loss = torch.mean(fusion_bce_loss)

            source_target_mmd_weight = segmented_function(e)
            target_distill_weight = distill_weight

            total_clf_loss = 1 * eeg_clf_loss / 4 + 1 * eye_clf_loss / 4 + 1 * fusion_clf_loss / 2
            total_cls_loss = 1 * eeg_cls_loss / 4 + 1 * eye_cls_loss / 4 + 1 * fusion_cls_loss / 2
            total_mmd_loss = 1 * eeg_mmd_loss / 4 + 1 * eye_mmd_loss / 4 + 1 * fusion_mmd_loss / 2
            total_distill_loss = 1 * eeg_distill_loss / 2 + 1 * eye_distill_loss / 2

            if total_clf_loss <= 0.1:
                target_fusion_const_weight = 0.5
            elif 0.1 < total_clf_loss < 0.2:
                target_fusion_const_weight = 0.2
            else:
                target_fusion_const_weight = 0.1

            loss = (total_clf_loss + cls_weight * total_cls_loss + source_target_mmd_weight * total_mmd_loss
                    + target_fusion_const_weight * fusion_target_clf_loss + target_distill_weight * total_distill_loss)

            loss.backward()
            optimizer.step()

            # if mlen % 10 == 0:
            #     model.visualization(eeg_target, eye_target, label_target, 'SEED')

            train_loss_clf.update(total_clf_loss.item())
            train_loss_cls.update(total_cls_loss.item())
            train_loss_align_mmd.update(total_mmd_loss.item())
            train_loss_align_single.update(total_distill_loss.item())
            train_loss_align_fusion.update(fusion_target_clf_loss.item())
            train_loss_total.update(loss.item())

        # Test
        acc, prediction, conf_matrix = evaluate(model, target_test_loader)
        log.append([e, train_loss_clf.avg, train_loss_cls.avg, train_loss_align_mmd.avg, train_loss_align_single.avg,
                    train_loss_align_fusion.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt(f'./train_log_seed_subject.csv', np_log, delimiter=',', fmt='%.6f')
        
        # print('Epoch: [{:2d}/{}], loss_clf: {:.4f}, loss_cls: {:.4f}, loss_align_mmd: {:.4f}, '
        #       'loss_align_single: {:.4f}, loss_align_fusion: {:.4f}, loss_total: {:.4f}, acc: {:.4f}'
        #       .format(e, args.n_epoch, train_loss_clf.avg, train_loss_cls.avg, train_loss_align_mmd.avg,
        #               train_loss_align_single.avg, train_loss_align_fusion.avg, train_loss_total.avg, acc))

        if best_acc < acc:
            best_acc = acc
            best_confusion_matrix = conf_matrix
            torch.save(model.state_dict(), model_filename)

    print('Transfer result: {:.4f}'.format(best_acc))
    print(best_confusion_matrix)

    return best_acc, best_confusion_matrix

def main(session, layer_num, distill_weight, cls_weight, seed):
    setup_seed(seed)
    all_test_results = []
    all_epochs_pseudo_labels = []
    final_confusion_matrix = None

    for i in range(12):
        print('test_id = ', i + 1)

        batch_size = args.batch_size
        source_loader, target_train_loader, target_test_loader = load_data(test_id=i, BATCH_SIZE=batch_size, session=session)
        model = graph.UF_AMA(args.n_class, transfer_loss=args.trans_loss, base_net=args.model, layer_num=layer_num).to(DEVICE)

        optimizer = torch.optim.Adam([
            {'params': model.eeg_base_network.parameters()},
            {'params': model.eye_base_network.parameters()},
            {'params': model.fusion_base_network.parameters()}
        ], lr=args.lr)

        model.init_all()
        best_acc, best_confusion_matrix = train(source_loader, target_train_loader, target_test_loader, model, optimizer, cls_weight, distill_weight, i + 1)
        all_test_results.append(best_acc)
        
        if final_confusion_matrix is None:
            final_confusion_matrix = best_confusion_matrix
        else:
            final_confusion_matrix += best_confusion_matrix

    stacked_results = torch.stack(all_test_results)
    average_result = torch.mean(stacked_results)

    print("all results:", stacked_results)
    print("Average result:", average_result)
    
    return final_confusion_matrix


if __name__ == '__main__':
    
    combined_confusion_matrix = None

    for session in [1, 2, 3]:
        print("\nsession: " + str(session))
        final_confusion_matrix = main(session, 1, 1e-5, 1, 50)
        if combined_confusion_matrix is None:
            combined_confusion_matrix = final_confusion_matrix
        else:
            combined_confusion_matrix += final_confusion_matrix
    
    plt.rcParams['font.family'] = 'Times New Roman'
    class_names = ['negative', 'neutral', 'positive']
    cm_normalized = combined_confusion_matrix.astype('float') / combined_confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6.5))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={'size': 18, 'weight': 'bold'})

    ax.tick_params(axis='both', labelsize=18)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Times New Roman')
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.savefig('seed_subject_confusion_matrix.pdf', bbox_inches='tight')
