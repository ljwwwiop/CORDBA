export CUDA_VISIBLE_DEVICES=3



## PN-ModelNet40
python defense_irba_attack.py --dataset modelnet40 --num_category 40 --model pointnet_cls --poisoned_rate 0.1 --target_label 35 --num_anchor 16 --R_alpha 5 --S_size 5 --process_data --use_uniform_sample --gpu 1 \
    --checkpoint_path './log/modelnet40_pointnet_cls/5.0_5.0_16_0.1/irba-baseline-40/checkpoints/last_model.pth' \
    --recon_model_path './checkpoint/2024/modelnet40-best-rec/best_parameters.tar' \
    --z_dim 1024


## PN-ShapePartNet
python defense_irba_attack.py --dataset shapenet --num_category 16 --model pointnet_cls --poisoned_rate 0.1 --target_label 8 --num_anchor 16 --R_alpha 5 --S_size 5 --process_data --use_uniform_sample --gpu 1 \
    --checkpoint_path './log/shapenet_pointnet_cls/5.0_5.0_16_0.1/2024-10-19_08-06/checkpoints/last_model.pth' \
    --recon_model_path './checkpoint/2024/rec-shape-dim-512/best_parameters.tar' \
    --z_dim 512

