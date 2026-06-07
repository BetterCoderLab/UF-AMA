import torch.nn as nn
import torch.nn.functional as F

import backbone
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

import loss_mmd
import loss_cmmd

GAMMA = 10 ^ 3

class UF_AMA(nn.Module):

    def __init__(self, num_class, base_net='transformer', transfer_loss='mmd', use_bottleneck=False, width=32,
                 layer_num=1):
        super(UF_AMA, self).__init__()
        if num_class == 3:
            self.eeg_base_network = backbone.network_dict[base_net](is_eeg=True, layer_num=layer_num)
            self.eye_base_network = backbone.network_dict[base_net](is_eeg=False, layer_num=layer_num, peri_dim=33)
            self.fusion_base_network = backbone.network_dict['fusion'](peri_dim=33)
        elif num_class == 4:
            self.eeg_base_network = backbone.network_dict[base_net](is_eeg=True, layer_num=layer_num)
            self.eye_base_network = backbone.network_dict[base_net](is_eeg=False, layer_num=layer_num, peri_dim=31)
            self.fusion_base_network = backbone.network_dict['fusion'](peri_dim=31)

        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss

        eeg_classifier_layer_list = [nn.Linear(width * 2, width), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(width, num_class)]
        eye_classifier_layer_list = [nn.Linear(width * 2, width), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(width, num_class)]
        fusion_classifier_layer_list = [nn.Linear(width * 8, width * 2), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width * 2, num_class)]

        self.eeg_classifier = nn.Sequential(*eeg_classifier_layer_list)
        self.eye_classifier = nn.Sequential(*eye_classifier_layer_list)
        self.fusion_classifier = nn.Sequential(*fusion_classifier_layer_list)
        self.softmax = nn.Softmax(dim=1)
        self.num_class = num_class
        self.base_net = base_net

    def forward(self, e, eeg_source, eye_source, eeg_target, eye_target, s_label):
        if self.base_net == 'transformer':
            eeg_source = eeg_source.reshape(eeg_source.shape[0], 62, 5)
            eeg_target = eeg_target.reshape(eeg_target.shape[0], 62, 5)
            eye_source = eye_source.reshape(eye_source.shape[0], eye_source.shape[1], 1)
            eye_target = eye_target.reshape(eye_target.shape[0], eye_target.shape[1], 1)

        # Feature extractor
        eeg_source_2D_data, eeg_source_1D_data = self.eeg_base_network(eeg_source)
        eeg_target_2D_data, eeg_target_1D_data = self.eeg_base_network(eeg_target)
        eye_source_2D_data, eye_source_1D_data = self.eye_base_network(eye_source)
        eye_target_2D_data, eye_target_1D_data = self.eye_base_network(eye_target)

        # Fusion
        fusion_source_data = self.fusion_base_network(eeg_source_2D_data, eye_source_2D_data)
        fusion_target_data = self.fusion_base_network(eeg_target_2D_data, eye_target_2D_data)

        # Classifier
        eeg_source_clf = self.eeg_classifier(eeg_source_1D_data)
        eeg_target_clf = self.eeg_classifier(eeg_target_1D_data)
        eye_source_clf = self.eye_classifier(eye_source_1D_data)
        eye_target_clf = self.eye_classifier(eye_target_1D_data)
        fusion_source_clf = self.fusion_classifier(fusion_source_data)

        # Softmax
        eeg_source_sft = self.softmax(eeg_source_clf)
        eeg_target_sft = self.softmax(eeg_target_clf)
        eye_source_sft = self.softmax(eye_source_clf)
        eye_target_sft = self.softmax(eye_target_clf)
        fusion_source_sft = self.softmax(fusion_source_clf)

        eeg_sim_matrix = self.get_cos_similarity_distance(eeg_source_sft)
        eye_sim_matrix = self.get_cos_similarity_distance(eye_source_sft)
        fusion_sim_matrix = self.get_cos_similarity_distance(fusion_source_sft)
        estimated_sim_truth = self.get_cos_similarity_distance(s_label)

        # Target filter
        eeg_target_max_probability, eeg_target_pseudo_labels = torch.max(eeg_target_sft, dim=1)
        eye_target_max_probability, eye_target_pseudo_labels = torch.max(eye_target_sft, dim=1)

        eeg_modality_id = torch.zeros(len(eeg_target_pseudo_labels), dtype=torch.long, device=eeg_target_pseudo_labels.device)
        eye_modality_id = torch.ones(len(eye_target_pseudo_labels), dtype=torch.long, device=eye_target_pseudo_labels.device)
        eeg_trial_id = torch.arange(len(eeg_target_pseudo_labels), dtype=torch.long, device=eeg_target_pseudo_labels.device)
        eye_trial_id = torch.arange(len(eye_target_pseudo_labels), dtype=torch.long, device=eye_target_pseudo_labels.device)

        all_similarity_max_score = torch.cat([eeg_target_max_probability, eye_target_max_probability])
        all_pseudo_label = torch.cat([eeg_target_pseudo_labels, eye_target_pseudo_labels])
        all_data_1D = torch.cat([eeg_target_1D_data, eye_target_1D_data])
        all_modality_id = torch.cat([eeg_modality_id, eye_modality_id])
        all_trial_id = torch.cat([eeg_trial_id, eye_trial_id])

        score1, score2 = eeg_target_max_probability, eye_target_max_probability
        selected_index = torch.topk(all_similarity_max_score, k=score1.shape[0])[1]
        other_index = torch.topk(all_similarity_max_score, k=score2.shape[0], largest=False)[1]
        mid_confidence_score = torch.topk(all_similarity_max_score, k=score1.shape[0])[0][-1]

        filtered = self.target_filter(all_trial_id, all_modality_id, all_pseudo_label, all_data_1D,
                                      eeg_target_2D_data, eye_target_2D_data, selected_index, other_index)

        ## Loss calculation
        criterion = torch.nn.CrossEntropyLoss()

        if mid_confidence_score.item() > 0.9:
            # EYE branch -- EEG teach EYE
            if filtered['distill_eeg'] is not None:
                # Attention-selected pseudo labels
                fusion_target_filtered_data = self.fusion_base_network(filtered['distill_eeg']['eeg_features_2D'],
                                                                       filtered['distill_eeg']['eye_features_2D'])
                fusion_target_clf = self.fusion_classifier(fusion_target_filtered_data)
                fusion_target_pseudo_labels = torch.max(self.softmax(fusion_target_clf), dim=1)[1]
                # High-quality pseudo-label delivery and pseudo-label comparison
                eye_target_mask = fusion_target_pseudo_labels == filtered['distill_eeg']['labels']
                eye_target_clf = self.eye_classifier(filtered['distill_eeg']['eye_features'])[eye_target_mask]
                eye_target_pseudo_labels = filtered['distill_eeg']['labels'][eye_target_mask]
                if len(eye_target_mask) < 3 or torch.all(eye_target_mask == False):
                    eeg_distill_loss = torch.tensor(0.0, dtype=eeg_source_1D_data.dtype,
                                                    device=eeg_source_1D_data.device)
                else:
                    eeg_distill_loss = criterion(eye_target_clf, eye_target_pseudo_labels)
            else:
                eeg_distill_loss = torch.tensor(0.0, dtype=eeg_source_1D_data.dtype, device=eeg_source_1D_data.device)

            # EEG branch -- EYE teach EEG
            if filtered['distill_eye'] is not None:
                # Attention-selected pseudo labels
                fusion_target_filtered_data = self.fusion_base_network(filtered['distill_eye']['eeg_features_2D'],
                                                                       filtered['distill_eye']['eye_features_2D'])
                fusion_target_clf = self.fusion_classifier(fusion_target_filtered_data)
                fusion_target_pseudo_labels = torch.max(self.softmax(fusion_target_clf), dim=1)[1]
                # High-quality pseudo-label delivery and pseudo-label comparison
                eeg_target_mask = fusion_target_pseudo_labels == filtered['distill_eye']['labels']
                eeg_target_clf = self.eeg_classifier(filtered['distill_eye']['eeg_features'])[eeg_target_mask]
                eeg_target_pseudo_labels = filtered['distill_eye']['labels'][eeg_target_mask]
                if len(eeg_target_mask) < 3 or torch.all(eeg_target_mask == False):
                    eye_distill_loss = torch.tensor(0.0, dtype=eye_source_1D_data.dtype,
                                                    device=eye_source_1D_data.device)
                else:
                    eye_distill_loss = criterion(eeg_target_clf, eeg_target_pseudo_labels)
            else:
                eye_distill_loss = torch.tensor(0.0, dtype=eye_source_1D_data.dtype, device=eye_source_1D_data.device)
        else:
            eeg_distill_loss = torch.tensor(0.0, dtype=eeg_source_1D_data.dtype, device=eeg_source_1D_data.device)
            eye_distill_loss = torch.tensor(0.0, dtype=eye_source_1D_data.dtype, device=eye_source_1D_data.device)

        eeg_mmd_loss = self.adapt_loss(eeg_source_1D_data, eeg_target_1D_data, self.transfer_loss)
        eye_mmd_loss = self.adapt_loss(eye_source_1D_data, eye_target_1D_data, self.transfer_loss)
        fusion_mmd_loss = self.adapt_loss(fusion_source_data, fusion_target_data, self.transfer_loss)

        if filtered['filtered_target'] is not None:
            fusion_target_filtered_data = self.fusion_base_network(filtered['filtered_target']['eeg_features_2D'],
                                                                   filtered['filtered_target']['eye_features_2D'])
            fusion_target_clf = self.fusion_classifier(fusion_target_filtered_data)
            fusion_target_pseudo_labels = filtered['filtered_target']['labels']
            fusion_target_clf_loss = criterion(fusion_target_clf, fusion_target_pseudo_labels)
        else:
            fusion_target_clf_loss = torch.tensor(0.0, dtype=fusion_source_data.dtype, device=fusion_source_data.device)

        return (eeg_source_clf, eeg_mmd_loss, eeg_distill_loss, eeg_sim_matrix,
                eye_source_clf, eye_mmd_loss, eye_distill_loss, eye_sim_matrix,
                fusion_source_clf, fusion_mmd_loss, fusion_target_clf_loss, fusion_sim_matrix, estimated_sim_truth)

    def init_all(self):
        self.eeg_base_network.init_weights()
        self.eye_base_network.init_weights()
        self.fusion_base_network.init_weights()
        for m in self.eeg_classifier.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()
        for m in self.eye_classifier.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()
        for m in self.fusion_classifier.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()

    def predict(self, eeg, eye):
        if self.base_net == 'transformer':
            eeg = eeg.reshape(eeg.shape[0], 62, 5)
            eye = eye.reshape(eye.shape[0], eye.shape[1], 1)

        eeg_2D, _ = self.eeg_base_network(eeg)
        eye_2D, _ = self.eye_base_network(eye)

        fusion = self.fusion_base_network(eeg_2D, eye_2D)
        fusion_clf = self.fusion_classifier(fusion)

        return fusion_clf

    def target_filter(self, all_trial_id, all_modality_id, all_pseudo_label, all_data, eeg_data_2D, eye_data_2D,
                      selected_index, other_index):
        selected_pseudo_label = all_pseudo_label[selected_index]
        selected_modality_id = all_modality_id[selected_index]
        selected_trial_id = all_trial_id[selected_index]
        selected_data_1D = all_data[selected_index]

        other_pseudo_label = all_pseudo_label[other_index]
        other_modality_id = all_modality_id[other_index]
        other_trial_id = all_trial_id[other_index]
        other_data_1D = all_data[other_index]

        selected_trial_dict = dict()
        other_trial_dict = dict()
        result = dict()

        for i in range(len(selected_trial_id)):
            trial = selected_trial_id[i].item()
            modality = selected_modality_id[i].item()
            if trial not in selected_trial_dict:
                selected_trial_dict[trial] = {'eeg': dict(), 'eye': dict()}
            if modality == 0:
                selected_trial_dict[trial]['eeg'] = {'data': selected_data_1D[i], 'data_2D': eeg_data_2D[trial],
                                                     'label': selected_pseudo_label[i].item()}
            else:
                selected_trial_dict[trial]['eye'] = {'data': selected_data_1D[i], 'data_2D': eye_data_2D[trial],
                                                     'label': selected_pseudo_label[i].item()}
        for i in range(len(other_trial_id)):
            trial = other_trial_id[i].item()
            modality = other_modality_id[i].item()
            if trial not in other_trial_dict:
                other_trial_dict[trial] = {'eeg': dict(), 'eye': dict()}
            if modality == 0:
                other_trial_dict[trial]['eeg'] = {'data': other_data_1D[i], 'data_2D': eeg_data_2D[trial],
                                                  'label': other_pseudo_label[i].item()}
            else:
                other_trial_dict[trial]['eye'] = {'data': other_data_1D[i], 'data_2D': eye_data_2D[trial],
                                                  'label': other_pseudo_label[i].item()}

        distill_eeg_dict = {'eeg_data': list(), 'eye_data': list(), 'eeg_data_2D': list(), 'eye_data_2D': list(), 'label': list()}
        distill_eye_dict = {'eeg_data': list(), 'eye_data': list(), 'eeg_data_2D': list(), 'eye_data_2D': list(), 'label': list()}
        filtered_target_dict = {'eeg_data': list(), 'eye_data': list(), 'eeg_data_2D': list(), 'eye_data_2D': list(), 'label': list()}

        for trial, info in selected_trial_dict.items():
            has_eeg = len(info['eeg']) > 0
            has_eye = len(info['eye']) > 0
            if has_eeg and has_eye:
                if info['eeg']['label'] == info['eye']['label']:
                    filtered_target_dict['eeg_data'].append(info['eeg']['data'])
                    filtered_target_dict['eye_data'].append(info['eye']['data'])
                    filtered_target_dict['eeg_data_2D'].append(info['eeg']['data_2D'])
                    filtered_target_dict['eye_data_2D'].append(info['eye']['data_2D'])
                    filtered_target_dict['label'].append(info['eeg']['label'])
                else:
                    continue
            elif has_eeg and trial in other_trial_dict.keys() and 'data' in other_trial_dict[trial]['eye'].keys():
                distill_eeg_dict['eeg_data'].append(info['eeg']['data'])
                distill_eeg_dict['eye_data'].append(other_trial_dict[trial]['eye']['data'])
                distill_eeg_dict['eeg_data_2D'].append(info['eeg']['data_2D'])
                distill_eeg_dict['eye_data_2D'].append(other_trial_dict[trial]['eye']['data_2D'])
                distill_eeg_dict['label'].append(info['eeg']['label'])
            elif has_eye and trial in other_trial_dict.keys() and 'data' in other_trial_dict[trial]['eeg'].keys():
                distill_eye_dict['eeg_data'].append(other_trial_dict[trial]['eeg']['data'])
                distill_eye_dict['eye_data'].append(info['eye']['data'])
                distill_eye_dict['eeg_data_2D'].append(other_trial_dict[trial]['eeg']['data_2D'])
                distill_eye_dict['eye_data_2D'].append(info['eye']['data_2D'])
                distill_eye_dict['label'].append(info['eye']['label'])

        if len(distill_eeg_dict['label']):
            result['distill_eeg'] = {
                'eeg_features': torch.stack(distill_eeg_dict['eeg_data']),
                'eye_features': torch.stack(distill_eeg_dict['eye_data']),
                'eeg_features_2D': torch.stack(distill_eeg_dict['eeg_data_2D']),
                'eye_features_2D': torch.stack(distill_eeg_dict['eye_data_2D']),
                'labels': torch.tensor(distill_eeg_dict['label'], dtype=torch.long, device=all_data.device)
            }
        else:
            result['distill_eeg'] = None
        if len(distill_eye_dict['label']):
            result['distill_eye'] = {
                'eeg_features': torch.stack(distill_eye_dict['eeg_data']),
                'eye_features': torch.stack(distill_eye_dict['eye_data']),
                'eeg_features_2D': torch.stack(distill_eye_dict['eeg_data_2D']),
                'eye_features_2D': torch.stack(distill_eye_dict['eye_data_2D']),
                'labels': torch.tensor(distill_eye_dict['label'], dtype=torch.long, device=all_data.device)
            }
        else:
            result['distill_eye'] = None
        if len(filtered_target_dict['label']):
            result['filtered_target'] = {
                'eeg_features': torch.stack(filtered_target_dict['eeg_data']),
                'eye_features': torch.stack(filtered_target_dict['eye_data']),
                'eeg_features_2D': torch.stack(filtered_target_dict['eeg_data_2D']),
                'eye_features_2D': torch.stack(filtered_target_dict['eye_data_2D']),
                'labels': torch.tensor(filtered_target_dict['label'], dtype=torch.long, device=all_data.device)
            }
        else:
            result['filtered_target'] = None

        return result

    def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_distill_loss(self, data_a, data_b, label_matrix):
        temperature = 0.5
        epsilon = 0.37

        data_a = F.normalize(data_a, dim=1)
        data_b = F.normalize(data_b, dim=1)

        similarity_matrix = F.cosine_similarity(data_a.unsqueeze(1), data_b.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix / temperature

        nominator = label_matrix * torch.exp(similarity_matrix)
        denominator = torch.exp(similarity_matrix)

        if torch.sum(nominator) == 0 or torch.sum(denominator) == 0:
            return torch.tensor(0.0, device=data_a.device)

        nominator_sum = torch.sum(nominator, dim=1)
        denominator_sum = torch.sum(denominator, dim=1)

        loss_partial = -torch.log(torch.clamp(nominator_sum / denominator_sum, min=epsilon, max=1))
        loss = torch.mean(loss_partial)

        return loss

    def get_label_matrix(self, label_a, label_b):
        labels_i = label_a.unsqueeze(1)
        labels_j = label_b.unsqueeze(0)
        label_matrix = (labels_i == labels_j).float()
        return label_matrix

    def adapt_loss(self, X, Y, adapt_loss):
        # loss = mmd.MMD_loss(X, Y)
        loss = loss_mmd.mmd_rbf_accelerate(X, Y)
        # loss,_,_ = mk_mmd.mix_rbf_mmd2_and_ratio(X, Y, [GAMMA])
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        # if adapt_loss == 'mmd':
        #     mmd_loss = mmd.MMD_loss()
        #     loss = mmd_loss(X, Y)
        # elif adapt_loss == 'coral':
        #     loss = CORAL(X, Y)
        # else:
        #     loss = 0
        return loss

    def visualization(self, eeg_features, eye_features, labels, dataset):
        if self.base_net == 'transformer':
            eeg_features = eeg_features.reshape(eeg_features.shape[0], 62, 5)
            eye_features = eye_features.reshape(eye_features.shape[0], eye_features.shape[1], 1)

        eeg_2D, eeg_1D = self.eeg_base_network(eeg_features)
        eye_2D, eye_1D = self.eye_base_network(eye_features)
        fusion_features = self.fusion_base_network(eeg_2D, eye_2D)

        fusion_features_np = fusion_features.cpu().detach().numpy()
        labels_np = np.argmax(labels.cpu().detach().numpy(), axis=1)

        # 创建图形
        plt.figure(figsize=(10, 8))

        if dataset == 'SEED':
            colors1 = '#00CED1'  # 蓝绿色
            colors2 = '#DC143C'  # 深红色
            colors3 = '#008000'  # 绿色

            # 为类别定义更有意义的标签
            class_names = ['Negative', 'Neutral', 'Positive']

            fusion_features_tsne = TSNE(perplexity=10, n_components=2, init='pca', max_iter=300).fit_transform(
                fusion_features_np.astype('float32'))

            class_1 = fusion_features_tsne[np.where(labels_np == 0)[0]]
            class_2 = fusion_features_tsne[np.where(labels_np == 1)[0]]
            class_3 = fusion_features_tsne[np.where(labels_np == 2)[0]]

            plt.scatter(class_1[:, 0], class_1[:, 1], c='none', marker='o',
                        edgecolors=colors1, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[0])
            plt.scatter(class_2[:, 0], class_2[:, 1], c='none', marker='s',
                        edgecolors=colors2, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[1])
            plt.scatter(class_3[:, 0], class_3[:, 1], c='none', marker='^',
                        edgecolors=colors3, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[2])

        elif dataset == 'SEED-IV':
            colors1 = '#00CED1'  # 蓝绿色
            colors2 = '#DC143C'  # 深红色
            colors3 = '#008000'  # 绿色
            colors4 = '#FFD700'  # 黄色

            # 为类别定义更有意义的标签
            class_names = ['Neutral', 'Sad', 'Fear', 'Happy']

            fusion_features_tsne = TSNE(perplexity=10, n_components=2, init='pca', max_iter=300).fit_transform(
                fusion_features_np.astype('float32'))

            class_1 = fusion_features_tsne[np.where(labels_np == 0)[0]]
            class_2 = fusion_features_tsne[np.where(labels_np == 1)[0]]
            class_3 = fusion_features_tsne[np.where(labels_np == 2)[0]]
            class_4 = fusion_features_tsne[np.where(labels_np == 3)[0]]

            plt.scatter(class_1[:, 0], class_1[:, 1], c='none', marker='o',
                        edgecolors=colors1, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[0])
            plt.scatter(class_2[:, 0], class_2[:, 1], c='none', marker='s',
                        edgecolors=colors2, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[1])
            plt.scatter(class_3[:, 0], class_3[:, 1], c='none', marker='^',
                        edgecolors=colors3, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[2])
            plt.scatter(class_4[:, 0], class_4[:, 1], c='none', marker='v',
                        edgecolors=colors4, alpha=0.6, linewidths=0.8,
                        s=60, label=class_names[3])

        # 设置坐标轴标签
        plt.xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
        plt.ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')

        # 显示坐标尺度
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格线
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加标题
        plt.title(f'{dataset} - Feature Visualization (t-SNE)', fontsize=16, fontweight='bold', pad=20)

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.show()
