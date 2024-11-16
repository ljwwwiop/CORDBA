
export CUDA_VISIBLE_DEVICES=1



### evalution

## modelnet40
python defense_pcba_attack.py --target_model pointnet_cls --dataset ModelNet40 --attack_dir attack_pcba --output_dir model_pcba_attacked --src_label 8 \
    --recon_model_path './checkpoint/2024/modelnet40-best-rec/best_parameters.tar' \
    --z_dim 1024




## shapenet
python defense_pcba_attack.py --target_model pointnet_cls --dataset ShapeNetPart --attack_dir attack_pcba --output_dir model_pcba_attacked --src_label 4 \
    --recon_model_path './checkpoint/2024/rec-shape-dim-512/best_parameters.tar' \
    --z_dim 512

