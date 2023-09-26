import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    # torch.arange(bsz) -> 0~(bsz-1)
    value = torch.arange(bsz).long().cuda()
    # Tensor.index_copy_(dim, index, tensor)
    # if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of self
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


class TSC(nn.Module):
    def __init__(self, num_class=4, 
                 tr=1, tw=0.2, targeted=True, 
                 random=True, T=0.07, K=256*3*4, feature_size=512) -> None:
        super().__init__()
        self.targeted = targeted
        self.tr = tr
        self.tw = tw
        self.n_cls = num_class
        self.K = K
        self.T = T
        self.random = random
        self.feature_size = feature_size
        
        optimal_target = np.load('./optimal_{}_{}.npy'.format(num_class, self.feature_size))
        optimal_target_order = np.arange(self.n_cls)
        target_repeat = self.tr * np.ones(self.n_cls)
        
        optimal_target = torch.Tensor(optimal_target).float()
        target_repeat = torch.Tensor(target_repeat).long()
        optimal_target = torch.cat(
            [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)
        target_labels = torch.cat(
            [torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))],
            dim=0).long().unsqueeze(-1)
        self.register_buffer("optimal_target", optimal_target)
        self.register_buffer("optimal_target_unique", optimal_target[::self.tr, :].contiguous().transpose(0, 1))
        self.register_buffer("target_labels", target_labels)
        
        self.register_buffer("queue", torch.randn(self.feature_size, K))
        self.register_buffer("queue_labels", -torch.ones(1, K).long())
        self.register_buffer("class_centroid", torch.randn(self.n_cls, self.feature_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.class_centroid = F.normalize(self.class_centroid, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue.shape[-1]:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[:, ptr:ptr + batch_size] = labels.T
        else:
            # print('ptr', ptr) # ptr 12032
            # print('self.queue.shape[-1]', self.queue.shape[-1]) # self.queue.shape[-1] 12288
            # print('self.queue.shape[-1]-ptr', self.queue.shape[-1]-ptr) # self.queue.shape[-1]-ptr 256
            self.queue[:, ptr:] = (keys.T)[:, :self.queue.shape[-1]-ptr]
            self.queue_labels[:, ptr:] = (labels.T)[:, :self.queue.shape[-1]-ptr]
            self.queue[:, :ptr + batch_size-self.queue.shape[-1]] = (keys.T)[:, self.queue.shape[-1]-ptr:]
            self.queue_labels[:, :ptr + batch_size-self.queue.shape[-1]] = (labels.T)[:, self.queue.shape[-1]-ptr:]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def reshape(self, tensor):
        if len(tensor.shape)==3:
            tensor = tensor.view(-1)
        else:
            tensor = tensor.permute(0, 2, 3, 1)
            tensor = tensor.flatten(0, 2)
        return tensor
    
    
    def label_onehot(self, inputs):
        batch_size, im_h, im_w = inputs.shape
        # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
        # we will still mask out those invalid values in valid mask
        inputs = torch.relu(inputs).data.cpu().type(torch.int64)
        outputs = torch.zeros([batch_size, self.n_cls, im_h, im_w]).to(inputs.device)
        return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)
        
        
    def find_hard_sampling(self, prob, label_raw, rep_raw, rep_raw_t, threshold, num_queries=256, random=True):
        # prob: bs, num_class, 256, 256
        # label: bs, num_class, 256, 256
        # rep: bs, 16, 256, 256
        # bs, num_feat, h, w = rep_raw.shape
        device = rep_raw.device
        rep = rep_raw.permute(0, 2, 3, 1)
        rep_raw_t = rep_raw_t.permute(0, 2, 3, 1)
        if len(label_raw.shape)==3:
            label = self.label_onehot(label_raw)
        else:
            label = label_raw
        num_segments = label.shape[1]
        
        seg_num_list = []
        seg_feat_all_list = []
        seg_feat_all_t_list = []
        seg_feat_hard_list = []
        seg_feat_hard_t_list = []
        seg_label = []
        for i in range(num_segments):
            valid_pixel_seg = label[:, i]  # select binary mask for i-th class
            if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
                continue
            prob_seg = prob[:, i, :, :]
            seg_label.append(i)
            rep_mask_hard = (prob_seg.cpu() < threshold) * valid_pixel_seg.bool().cpu()  # select hard queries

            seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
            seg_feat_all_t_list.append(rep_raw_t[valid_pixel_seg.bool()])
            
            seg_feat_hard_list.append(rep[rep_mask_hard])
            seg_feat_hard_t_list.append(rep_raw_t[rep_mask_hard])
            
            seg_num_list.append(int(valid_pixel_seg.sum().item()))
        if len(seg_num_list) <= 1:
            rep_raw = self.reshape(rep_raw)
            rep_raw_t = self.reshape(rep_raw_t)
            label_ = self.reshape(label_raw).unsqueeze(-1).clone()
            # print('label_', label_.shape) # torch.Size([65536])
            # print('rep_raw', rep_raw.shape) # torch.Size([65536, 16])
            idx = torch.randint(low=0, high=label_.shape[0], size=(num_queries,))
            # print('idx', idx.shape) # idx torch.Size([256])
            # print('min, max', idx.min(), idx.max())
            to_return_rep = rep_raw[idx]
            to_return_rep_t = rep_raw_t[idx]
            to_return_label = label_[idx].squeeze(-1)
            return to_return_rep, to_return_rep_t, to_return_label
        else:
            all_feats = []
            all_feats_t = []
            all_labels = []
            if not random:
                valid_seg = len(seg_num_list)
                for i in range(valid_seg):
                    if len(seg_feat_hard_list[i]) > 0:
                        seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                        anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                        anchor_feat_hard_t = seg_feat_hard_t_list[i][seg_hard_idx]
                        all_feats.append(anchor_feat_hard)
                        all_feats_t.append(anchor_feat_hard_t)
                        cur_label = torch.ones(num_queries, dtype=torch.long, device=device)*seg_label[i]
                        all_labels.append(cur_label)
                    else:
                        continue
            else:
                valid_seg = len(seg_num_list)
                for i in range(valid_seg):
                    if len(seg_feat_all_list[i]) > 0:
                        seg_hard_idx = torch.randint(len(seg_feat_all_list[i]), size=(num_queries,))
                        anchor_feat_hard = seg_feat_all_list[i][seg_hard_idx]
                        anchor_feat_hard_t = seg_feat_all_t_list[i][seg_hard_idx]
                        all_feats.append(anchor_feat_hard)
                        all_feats_t.append(anchor_feat_hard_t)
                        cur_label = torch.ones(num_queries, dtype=torch.long, device=device)*seg_label[i]
                        all_labels.append(cur_label)
                    else:
                        continue
                
            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_feats_t = torch.cat(all_feats_t)
            idx, _ = get_shuffle_ids(all_feats.shape[0])
            all_feats = all_feats[idx]
            all_labels = all_labels[idx]
            all_feats_t = all_feats_t[idx]
            return all_feats, all_feats_t, all_labels


    def forward(self, pred_s, feat_s_raw, feat_t_raw, target):
        ## pred_s, pred_t: bs, n_cls, 256, 256
        ## feat_s, feat_t: bs, n_feat, 256, 256
        ## target: bs, 256, 256
        probs_s = torch.softmax(pred_s, dim=1)
        feat_s_raw = F.normalize(feat_s_raw, dim=1)
        feat_t_raw = F.normalize(feat_t_raw, dim=1)
        feats_s, feats_t, labels_cur = self.find_hard_sampling(
            probs_s, 
            label_raw=target, 
            rep_raw=feat_s_raw, 
            rep_raw_t=feat_t_raw, 
            threshold=0.7, 
            num_queries=256, 
            random=self.random, 
        )
        labels_cur = labels_cur.view(-1, 1)
        # print('feats_s', feats_s.shape)
        # print('feats_t', feats_t.shape)
        l_pos = torch.einsum('nc,nc->n', [feats_s, feats_t]).unsqueeze(-1)
        if self.targeted:
            queue_negatives = self.queue.clone().detach()
            target_negatives = self.optimal_target.transpose(0, 1)
            l_neg = torch.einsum('nc,ck->nk', [feats_s, torch.cat([queue_negatives, target_negatives], dim=1)])
        else:
            # print('feats_s', feats_s.shape) # feats_s torch.Size([512, 256])
            # print('self.queue', self.queue.shape) # self.queue torch.Size([512, 3072])
            l_neg = torch.einsum('nc,ck->nk', [feats_s, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        if self.targeted:
            queue_labels = self.queue_labels.clone().detach()
            target_labels = self.target_labels.transpose(0, 1)
            
            with torch.no_grad():
                mask = torch.eq(labels_cur, torch.cat([queue_labels, torch.full_like(target_labels, -1)], dim=1)).float()
                im_labels_all = labels_cur
                features_all = feats_s.detach()
                # update memory bank class centroids
                for one_label in torch.unique(im_labels_all):
                    class_centroid_batch = F.normalize(torch.mean(features_all[im_labels_all[:, 0].eq(one_label), :], dim=0), dim=0)
                    self.class_centroid[one_label] = 0.9*self.class_centroid[one_label] + 0.1*class_centroid_batch
                    self.class_centroid[one_label] = F.normalize(self.class_centroid[one_label], dim=0)
                centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.optimal_target_unique])
                centroid_target_dist = centroid_target_dist.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)
                
                for one_label, one_idx in zip(row_ind, col_ind):
                    if one_label not in labels_cur:
                        continue
                    one_indices = torch.Tensor([i+one_idx*self.tr for i in range(self.tr)]).long()
                    tmp = mask[labels_cur[:, 0].eq(one_label), :]
                    tmp[:, queue_labels.size(1)+one_indices] = 1
                    mask[labels_cur[:, 0].eq(one_label), :] = tmp
                
            mask_target = mask.clone()
            mask_target[:, :queue_labels.size(1)] = 0
            mask[:, queue_labels.size(1):] = 0
        else:
            mask = torch.eq(labels_cur, self.queue_labels.clone().detach()).float()
            
        # mask_pos_view = torch.zeros_like(mask)
        mask_pos_view = mask.clone()
        if self.targeted:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_target.clone()
            mask_pos_view += mask_target
        else:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_pos_view.clone()
            mask_pos_view_class[:, self.queue_labels.size(1):] = 0
            mask_pos_view_target[:, :self.queue_labels.size(1)] = 0
        
        mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).to(mask_pos_view.device), 
                                    mask_pos_view], dim=1)
        mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1])
                                            .to(mask_pos_view_class.device), mask_pos_view_class], dim=1)
        mask_pos_view_target = torch.cat([torch.zeros([mask_pos_view_target.shape[0], 1])
                                            .to(mask_pos_view_target.device), mask_pos_view_target], dim=1)
        
        logits /= self.T
        
        log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
        # print('mask_pos_view', mask_pos_view.shape)
        # print('mask_pos_view_class',mask_pos_view_class.shape) # mask_pos_view_class torch.Size([512, 6145])
        # print('log_prob', log_prob.shape) # log_prob torch.Size([512, 6149])
        loss_class = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        loss_target = - torch.sum((mask_pos_view_target * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        
        loss_target = loss_target * self.tw
        self._dequeue_and_enqueue(feats_t, labels_cur)
        
        return loss_class+loss_target
    
    
class TSC_3d(nn.Module):
    def __init__(self, num_class=4, 
                 tr=1, tw=0.2, targeted=True, 
                 random=True, T=0.07, K=256*3*4, feature_size=512) -> None:
        super().__init__()
        self.targeted = targeted
        self.tr = tr
        self.tw = tw
        self.n_cls = num_class
        self.K = K
        self.T = T
        self.random = random
        self.feature_size = feature_size
        
        optimal_target = np.load('./optimal_{}_{}.npy'.format(num_class, self.feature_size))
        optimal_target_order = np.arange(self.n_cls)
        target_repeat = self.tr * np.ones(self.n_cls)
        
        optimal_target = torch.Tensor(optimal_target).float()
        target_repeat = torch.Tensor(target_repeat).long()
        optimal_target = torch.cat(
            [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)
        target_labels = torch.cat(
            [torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))],
            dim=0).long().unsqueeze(-1)
        self.register_buffer("optimal_target", optimal_target)
        self.register_buffer("optimal_target_unique", optimal_target[::self.tr, :].contiguous().transpose(0, 1))
        self.register_buffer("target_labels", target_labels)
        
        self.register_buffer("queue", torch.randn(self.feature_size, K))
        self.register_buffer("queue_labels", -torch.ones(1, K).long())
        self.register_buffer("class_centroid", torch.randn(self.n_cls, self.feature_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.class_centroid = F.normalize(self.class_centroid, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue.shape[-1]:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[:, ptr:ptr + batch_size] = labels.T
        else:
            # print('ptr', ptr) # ptr 12032
            # print('self.queue.shape[-1]', self.queue.shape[-1]) # self.queue.shape[-1] 12288
            # print('self.queue.shape[-1]-ptr', self.queue.shape[-1]-ptr) # self.queue.shape[-1]-ptr 256
            self.queue[:, ptr:] = (keys.T)[:, :self.queue.shape[-1]-ptr]
            self.queue_labels[:, ptr:] = (labels.T)[:, :self.queue.shape[-1]-ptr]
            self.queue[:, :ptr + batch_size-self.queue.shape[-1]] = (keys.T)[:, self.queue.shape[-1]-ptr:]
            self.queue_labels[:, :ptr + batch_size-self.queue.shape[-1]] = (labels.T)[:, self.queue.shape[-1]-ptr:]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def reshape(self, tensor):
        if len(tensor.shape)==3:
            tensor = tensor.view(-1)
        else:
            tensor = tensor.permute(0, 2, 3, 4, 1)
            tensor = tensor.flatten(0, 3)
        return tensor
    
    
    def label_onehot(self, inputs):
        batch_size, im_h, im_w, im_d = inputs.shape
        # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
        # we will still mask out those invalid values in valid mask
        inputs = torch.relu(inputs).data.cpu().type(torch.int64)
        outputs = torch.zeros([batch_size, self.n_cls, im_h, im_w, im_d]).to(inputs.device)
        return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)
        
        
    def find_hard_sampling(self, prob, label_raw, rep_raw, rep_raw_t, threshold, num_queries=256, random=True):
        # prob: bs, num_class, 256, 256
        # label: bs, num_class, 256, 256
        # rep: bs, 16, 256, 256
        # bs, num_feat, h, w = rep_raw.shape
        device = rep_raw.device
        rep = rep_raw.permute(0, 2, 3, 4, 1)
        rep_raw_t = rep_raw_t.permute(0, 2, 3, 4, 1)
        if len(label_raw.shape)==4:
            label = self.label_onehot(label_raw)
        else:
            label = label_raw
        # print('label', label.shape) # label torch.Size([2, 112, 112, 80])
        num_segments = label.shape[1]
        
        seg_num_list = []
        seg_feat_all_list = []
        seg_feat_all_t_list = []
        seg_feat_hard_list = []
        seg_feat_hard_t_list = []
        seg_label = []
        for i in range(num_segments):
            valid_pixel_seg = label[:, i]  # select binary mask for i-th class
            if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
                continue
            prob_seg = prob[:, i, :, :]
            seg_label.append(i)
            rep_mask_hard = (prob_seg.cpu() < threshold) * valid_pixel_seg.bool().cpu()  # select hard queries

            seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
            seg_feat_all_t_list.append(rep_raw_t[valid_pixel_seg.bool()])
            
            seg_feat_hard_list.append(rep[rep_mask_hard])
            seg_feat_hard_t_list.append(rep_raw_t[rep_mask_hard])
            
            seg_num_list.append(int(valid_pixel_seg.sum().item()))
        if len(seg_num_list) <= 1:
            rep_raw = self.reshape(rep_raw)
            rep_raw_t = self.reshape(rep_raw_t)
            label_ = self.reshape(label_raw).unsqueeze(-1).clone()
            # print('label_', label_.shape) # torch.Size([65536])
            # print('rep_raw', rep_raw.shape) # torch.Size([65536, 16])
            idx = torch.randint(low=0, high=label_.shape[0], size=(num_queries,))
            # print('idx', idx.shape) # idx torch.Size([256])
            # print('min, max', idx.min(), idx.max())
            to_return_rep = rep_raw[idx]
            to_return_rep_t = rep_raw_t[idx]
            to_return_label = label_[idx].squeeze(-1)
            return to_return_rep, to_return_rep_t, to_return_label
        else:
            all_feats = []
            all_feats_t = []
            all_labels = []
            if not random:
                valid_seg = len(seg_num_list)
                for i in range(valid_seg):
                    if len(seg_feat_hard_list[i]) > 0:
                        seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                        anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                        anchor_feat_hard_t = seg_feat_hard_t_list[i][seg_hard_idx]
                        all_feats.append(anchor_feat_hard)
                        all_feats_t.append(anchor_feat_hard_t)
                        cur_label = torch.ones(num_queries, dtype=torch.long, device=device)*seg_label[i]
                        all_labels.append(cur_label)
                    else:
                        continue
            else:
                valid_seg = len(seg_num_list)
                for i in range(valid_seg):
                    if len(seg_feat_all_list[i]) > 0:
                        seg_hard_idx = torch.randint(len(seg_feat_all_list[i]), size=(num_queries,))
                        anchor_feat_hard = seg_feat_all_list[i][seg_hard_idx]
                        anchor_feat_hard_t = seg_feat_all_t_list[i][seg_hard_idx]
                        all_feats.append(anchor_feat_hard)
                        all_feats_t.append(anchor_feat_hard_t)
                        cur_label = torch.ones(num_queries, dtype=torch.long, device=device)*seg_label[i]
                        all_labels.append(cur_label)
                    else:
                        continue
                
            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_feats_t = torch.cat(all_feats_t)
            idx, _ = get_shuffle_ids(all_feats.shape[0])
            all_feats = all_feats[idx]
            all_labels = all_labels[idx]
            all_feats_t = all_feats_t[idx]
            return all_feats, all_feats_t, all_labels


    def forward(self, pred_s, feat_s_raw, feat_t_raw, target):
        ## pred_s, pred_t: bs, n_cls, 256, 256
        ## feat_s, feat_t: bs, n_feat, 256, 256
        ## target: bs, 256, 256
        probs_s = torch.softmax(pred_s, dim=1)
        feat_s_raw = F.normalize(feat_s_raw, dim=1)
        feat_t_raw = F.normalize(feat_t_raw, dim=1)
        feats_s, feats_t, labels_cur = self.find_hard_sampling(
            probs_s, 
            label_raw=target, 
            rep_raw=feat_s_raw, 
            rep_raw_t=feat_t_raw, 
            threshold=0.7, 
            num_queries=256, 
            random=self.random, 
        )
        labels_cur = labels_cur.view(-1, 1)
        # print('feats_s', feats_s.shape)
        # print('feats_t', feats_t.shape)
        l_pos = torch.einsum('nc,nc->n', [feats_s, feats_t]).unsqueeze(-1)
        if self.targeted:
            queue_negatives = self.queue.clone().detach()
            target_negatives = self.optimal_target.transpose(0, 1)
            l_neg = torch.einsum('nc,ck->nk', [feats_s, torch.cat([queue_negatives, target_negatives], dim=1)])
        else:
            # print('feats_s', feats_s.shape) # feats_s torch.Size([512, 256])
            # print('self.queue', self.queue.shape) # self.queue torch.Size([512, 3072])
            l_neg = torch.einsum('nc,ck->nk', [feats_s, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        if self.targeted:
            queue_labels = self.queue_labels.clone().detach()
            target_labels = self.target_labels.transpose(0, 1)
            
            with torch.no_grad():
                mask = torch.eq(labels_cur, torch.cat([queue_labels, torch.full_like(target_labels, -1)], dim=1)).float()
                im_labels_all = labels_cur
                features_all = feats_s.detach()
                # update memory bank class centroids
                for one_label in torch.unique(im_labels_all):
                    class_centroid_batch = F.normalize(torch.mean(features_all[im_labels_all[:, 0].eq(one_label), :], dim=0), dim=0)
                    self.class_centroid[one_label] = 0.9*self.class_centroid[one_label] + 0.1*class_centroid_batch
                    self.class_centroid[one_label] = F.normalize(self.class_centroid[one_label], dim=0)
                centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.optimal_target_unique])
                centroid_target_dist = centroid_target_dist.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)
                
                for one_label, one_idx in zip(row_ind, col_ind):
                    if one_label not in labels_cur:
                        continue
                    one_indices = torch.Tensor([i+one_idx*self.tr for i in range(self.tr)]).long()
                    tmp = mask[labels_cur[:, 0].eq(one_label), :]
                    tmp[:, queue_labels.size(1)+one_indices] = 1
                    mask[labels_cur[:, 0].eq(one_label), :] = tmp
                
            mask_target = mask.clone()
            mask_target[:, :queue_labels.size(1)] = 0
            mask[:, queue_labels.size(1):] = 0
        else:
            mask = torch.eq(labels_cur, self.queue_labels.clone().detach()).float()
            
        # mask_pos_view = torch.zeros_like(mask)
        mask_pos_view = mask.clone()
        if self.targeted:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_target.clone()
            mask_pos_view += mask_target
        else:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_pos_view.clone()
            mask_pos_view_class[:, self.queue_labels.size(1):] = 0
            mask_pos_view_target[:, :self.queue_labels.size(1)] = 0
        
        mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).to(mask_pos_view.device), 
                                    mask_pos_view], dim=1)
        mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1])
                                            .to(mask_pos_view_class.device), mask_pos_view_class], dim=1)
        mask_pos_view_target = torch.cat([torch.zeros([mask_pos_view_target.shape[0], 1])
                                            .to(mask_pos_view_target.device), mask_pos_view_target], dim=1)
        
        logits /= self.T
        
        log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
        # print('mask_pos_view', mask_pos_view.shape)
        # print('mask_pos_view_class',mask_pos_view_class.shape) # mask_pos_view_class torch.Size([512, 6145])
        # print('log_prob', log_prob.shape) # log_prob torch.Size([512, 6149])
        loss_class = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        loss_target = - torch.sum((mask_pos_view_target * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        
        loss_target = loss_target * self.tw
        self._dequeue_and_enqueue(feats_t, labels_cur)
        
        return loss_class+loss_target
    
    
if __name__ == '__main__':
    # tsc = TSC()
    # pred_s, feat_s_raw, feat_t_raw, target = \
    #     torch.randn(4, 4, 256, 256), \
    #     torch.randn(4, 512, 256, 256), \
    #     torch.randn(4, 512, 256, 256), \
    #     torch.ones(4, 256, 256).long()
    
    # target[:, :128, :] += 1
    # for i in range(5):
    #     tsc.targeted = i%2
    #     loss = tsc(pred_s, feat_s_raw, feat_t_raw, target)
    #     print(loss)
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np


    class uniform_loss(nn.Module):
        def __init__(self, t=0.07):
            super(uniform_loss, self).__init__()
            self.t = t

        def forward(self, x):
            return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()


    N = 2
    M = 128
    print("N =", N)
    print("M =", M)
    criterion = uniform_loss()
    x = Variable(torch.randn(N, M).float(), requires_grad=True)
    optimizer = optim.Adam([x], lr=1e-3)
    min_loss = 100
    optimal_target = None

    N_iter = 10000
    for i in range(N_iter):
        x_norm = F.normalize(x, dim=1)
        loss = criterion(x_norm)
        if i % 100 == 0:
            print(i, loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            optimal_target = x_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    np.save('optimal_{}_{}.npy'.format(N, M), optimal_target.detach().numpy())

    target = np.load(f'optimal_{N}_{M}.npy')
    print("optimal loss = ", criterion(torch.tensor(target)).item())

