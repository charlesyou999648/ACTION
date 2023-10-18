import torch
import logging
import sys
import os
import torchvision.models as models
from torchvision.utils import make_grid
import torch.optim as optim
import argparse
import shutil
import random
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import torch.utils.data.sampler as sampler
from tqdm import tqdm
from scipy.ndimage import zoom
from utils import losses, metrics, ramps

from model_3D import *
from augment_3d import *
from dataloaders.dataset import BaseDataSets, RandomGenerator
from build_dataset import BaseDataSetsWithIndex
from dataloaders.la_heart import LAHeartWithIndex, RandomRotFlip, RandomCrop, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/data/2018LA_Seg_Training/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/example_training', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')

parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')

parser.add_argument('--apply_aug', default='classmix', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--resume', type=str, default='ACDC/temperature_training_final', help='if we should resume from checkpoint') # 'ACDC/training_pool'
parser.add_argument('--K', type=int, default=36, help='the size of cache')               
parser.add_argument('--latent_pooling_size', type=int, default=1, help='the pooling size of latent vector')
parser.add_argument('--latent_feature_size', type=int, default=512, help='the feature size of latent vectors')
parser.add_argument('--output_pooling_size', type=int, default=8, help='the pooling size of output head')
parser.add_argument('--combinations', type=int, default=0, help='0: all, 1: no reco, 2: no unsup')

parser.add_argument('--temp_high', default=1.0, type=float)
parser.add_argument('--temp_low', default=0.1, type=float)

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train_l = LAHeartWithIndex(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(args.patch_size),
                          ToTensor(),
                          ]), index=args.labeled_num, label_type=1)

    db_train_u = LAHeartWithIndex(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(args.patch_size),
                          ToTensor(),
                          ]), index=args.labeled_num, label_type=0)

    while(len(db_train_l)<len(db_train_u)):
        db_train_l = torch.utils.data.ConcatDataset([db_train_l, db_train_l])

    train_l_loader = torch.utils.data.DataLoader(
            db_train_l,
            batch_size=batch_size,
            sampler=sampler.RandomSampler(data_source=db_train_l,
                                          replacement=True),
            drop_last=True,
        )

    train_u_loader = torch.utils.data.DataLoader(
            db_train_u,
            batch_size=batch_size,
            sampler=sampler.RandomSampler(data_source=db_train_u,
                                            replacement=True),
            drop_last=True,
        )

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_u_loader)))
    

    isd = ISD_3d(K=args.K, m=0.99, Ts=0, Tt=0, num_classes=num_classes, latent_pooling_size=args.latent_pooling_size, 
                latent_feature_size=args.latent_feature_size, output_pooling_size=args.output_pooling_size, 
                train_encoder=True, train_decoder=True, patch_size=20).cuda() # args.train_encoder # args.train_decoder

    isd.model.load_state_dict(torch.load("../model/{}_{}_labeled{}/{}/iter_30000.pth".format(
        args.resume, args.labeled_num, suffix, args.model), map_location=lambda storage, loc: storage))
    isd.ema_model.load_state_dict(torch.load("../model/{}_{}_labeled{}/{}/iter_30000.pth".format(
        args.resume, args.labeled_num, suffix, args.model), map_location=lambda storage, loc: storage))
    
    max_epoch = max_iterations // len(train_u_loader) + 1
    ema_model = isd.ema_model
    model = isd.model

    q_representation = RepresentationHead_3d(num_classes=128+64+32+16+16, output_channel=128).cuda()
    
    if torch.cuda.device_count() > 1:
        isd.data_parallel()
        q_representation = torch.nn.DataParallel(q_representation)
    isd.train()
    model.train()
    ema_model.train()
    q_representation.train()
    
    
    params = [p for p in model.parameters() if p.requires_grad]
    params_q = [p for p in q_representation.parameters() if p.requires_grad]
    optimizer = optim.SGD(params+params_q, lr=base_lr, weight_decay=0.0001, momentum=0.9, nesterov=True)
    
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    iter_num = 0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        train_l_dataset = iter(train_l_loader)
        train_u_dataset = iter(train_u_loader)

        for i in range(len(train_u_loader)):
            l_next = train_l_dataset.next()
            train_l_data, train_l_label = l_next['image'].cuda(), l_next['label'].cuda()
            u_next = train_u_dataset.next()
            train_u_data, _ = u_next['image'].cuda(), u_next['label'].cuda()
            
            
            pred_u, _ , _ = ema_model(train_u_data)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u, dim=1), dim=1) # torch.Size([6, 256, 256]) # torch.Size([6, 256, 256])

            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                (train_u_data, pseudo_labels, pseudo_logits)

            # apply mixing strategy: cutout, cutmix or classmix
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data_3d(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)

            # color jitter + gaussian blur
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, logits=train_u_aug_logits, scale_size=(1.0, 1.0), apply_augmentation=True)

            pred_l, _, l_feature_map = model(train_l_data)
            pred_u, _, u_feature_map = model(train_u_aug_data)

            for i in range(len(l_feature_map)):
                l_feature_map[i] = F.interpolate(l_feature_map[i], size=pred_l.shape[-3:], mode='trilinear', align_corners=True)
                u_feature_map[i] = F.interpolate(u_feature_map[i], size=pred_l.shape[-3:], mode='trilinear', align_corners=True)
                                
            l_feature_all = torch.concat(l_feature_map, dim=1)
            u_feature_all = torch.concat(u_feature_map, dim=1)

            rep_u = q_representation(u_feature_all)
            rep_l = q_representation(l_feature_all)

            rep_all = torch.cat((rep_l, rep_u))
            pred_all = torch.cat((pred_l, pred_u))

            
            outputs_soft = torch.softmax(pred_l, dim=1)
            loss_ce = ce_loss(pred_l, train_l_label.long())
            loss_dice = dice_loss(outputs_soft, train_l_label.unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            unsup_loss = compute_unsupervised_loss(pred_u, train_u_aug_label, train_u_aug_logits, args.strong_threshold)
            
            with torch.no_grad():
                train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float()
                mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')
                label_l = F.interpolate(label_onehot(train_l_label, args.num_classes), size=pred_all.shape[2:], mode='nearest')
                label_u = F.interpolate(label_onehot(train_u_aug_label, args.num_classes), size=pred_all.shape[2:], mode='nearest')
                label_all = torch.cat((label_l, label_u))

                prob_l = torch.softmax(pred_l, dim=1)
                prob_u = torch.softmax(pred_u, dim=1)
                prob_all = torch.cat((prob_l, prob_u))

            reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,
                                          args.temp, args.num_queries, args.num_negatives)
            
            loss = 0.01*reco_loss + unsup_loss + supervised_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            isd._momentum_update_key_encoder()
            iter_num+=1
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            ################################# val and output session #################

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/reco_loss', reco_loss, iter_num)
            writer.add_scalar('info/unsup_loss', unsup_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, reco_loss: %f, unsup_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), reco_loss.item(), unsup_loss.item(), ))

            if iter_num % 20 == 0:
                image = train_l_data[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                
                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)
                
                image = train_l_label[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)


            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(isd.model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"
        

def compute_unsupervised_loss(predict, target, logits, strong_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   
    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None, None] * loss, loss > 0))
    return weighted_loss


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w, im_d = inputs.shape
    inputs = torch.relu(inputs).data.cpu().type(torch.int64)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w, im_d]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def compute_reco_loss(rep, label, mask, prob, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h, im_d = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask.cpu()

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 4, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :, :]
        rep_mask_hard = (prob_seg.cpu() < strong_threshold) * valid_pixel_seg.bool().cpu()  # select hard queries

        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg

def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    torch.cuda.empty_cache()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    suffix = 'final'
    snapshot_path = "../model/{}_{}_labeled{}/{}".format(
        args.exp, args.labeled_num, suffix, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)