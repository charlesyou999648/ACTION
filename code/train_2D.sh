
# training action
# for pretraining stage 1
CUDA_VISIBLE_DEVICES=1 python train_2D_pretrain.py \
    --train_encoder 1 \
    --train_decoder 0 \
    --K 36 \
    --root_path /data/data/ACDC \
    --exp ACDC/pretrain \
    --k1 1 \
    --k2 1 \
    --latent_pooling_size 1 \
    --latent_feature_size 512 \
    --output_pooling_size 8 \
    --T_s 0.1 \
    --T_t 0.01 \
    --max_iterations 30000 \
    --labeled_num 1 \
    --num_classes 4 

# for pretraining stage 2
CUDA_VISIBLE_DEVICES=1 python train_2D_pretrain.py \
    --train_encoder 1 \
    --train_decoder 1 \
    --K 36 \
    --root_path /data/data/ACDC \
    --exp ACDC/pretrain \
    --resume ACDC/pretrain \
    --k1 1 \
    --k2 1 \
    --latent_pooling_size 1 \
    --latent_feature_size 512 \
    --output_pooling_size 8 \
    --T_s 0.1 \
    --T_t 0.01 \
    --max_iterations 30000 \
    --labeled_num 1 \
    --num_classes 4 

# for finetuning
CUDA_VISIBLE_DEVICES=1 python train_2D_action.py \
    --root_path /data/data/ACDC \
    --exp ACDC/action \
    --resume ACDC/pretrain \
    --batch_size 4 \
    --max_iterations 30000 \
    --apply_aug cutmix \
    --labeled_num 1 \
    --num_classes 4 


# for action++
# pretraining
CUDA_VISIBLE_DEVICES=1 python train_2D_pretrain++.py \
    --train_encoder 1 \
    --train_decoder 1 \
    --K 36 \
    --root_path /data/data/ACDC \
    --exp ACDC/pretrain++ \
    --k1 1 \
    --k2 1 \
    --latent_pooling_size 1 \
    --latent_feature_size 512 \
    --output_pooling_size 8 \
    --T_s 0.1 \
    --T_t 0.01 \
    --max_iterations 30000 \
    --labeled_num 1 \
    --num_classes 4 

# for finetuning
CUDA_VISIBLE_DEVICES=1 python train_2D_action++.py \
    --root_path /data/data/ACDC \
    --exp ACDC/action++ \
    --resume ACDC/pretrain++ \
    --batch_size 4 \
    --max_iterations 30000 \
    --apply_aug cutmix \
    --labeled_num 1 \
    --num_classes 4 