# training action
# for pretraining stage 1
CUDA_VISIBLE_DEVICES=1 python train_3D_pretrain.py \
    --train_encoder 1 \
    --train_decoder 0 \
    --K 36 \
    --exp LA/pretrain \
    --k1 1 \
    --k2 1 \
    --latent_pooling_size 1 \
    --latent_feature_size 128 \
    --output_pooling_size 4 \
    --T_s 0.1 \
    --T_t 0.6 \
    --max_iterations 30000 \
    --labeled_num 1 \
    --num_classes 2 


# for pretraining stage 2
CUDA_VISIBLE_DEVICES=1 python train_3D_pretrain.py \
    --train_encoder 1 \
    --train_decoder 1 \
    --K 36 \
    --exp LA/pretrain \
    --resume LA/pretrain \
    --k1 1 \
    --k2 1 \
    --latent_pooling_size 1 \
    --latent_feature_size 128 \
    --output_pooling_size 4 \
    --T_s 0.1 \
    --T_t 0.6 \
    --max_iterations 30000 \
    --labeled_num 1 \
    --num_classes 2 


# for finetuning
CUDA_VISIBLE_DEVICES=1 python train_3D_action.py \
    --resume LA/pretrain \
    --exp LA/action \
    --max_iterations 30000 \
    --labeled_num 1 \
    --batch_size 1 \
    --num_classes 2


# for action++
# pretraining
CUDA_VISIBLE_DEVICES=1 python train_3D_pretrain++.py \
    --train_encoder 1 \
    --train_decoder 1 \
    --K 36 \
    --exp LA/pretrain \
    --k1 1 \
    --k2 1 \
    --latent_pooling_size 1 \
    --latent_feature_size 128 \
    --output_pooling_size 4 \
    --T_s 0.1 \
    --T_t 0.6 \
    --max_iterations 30000 \
    --labeled_num 1 \
    --num_classes 2 

# for finetuning
CUDA_VISIBLE_DEVICES=1 python train_3D_action.py \
    --resume LA/pretrain++ \
    --exp LA/action++ \
    --max_iterations 30000 \
    --labeled_num 1 \
    --batch_size 1 \
    --num_classes 2