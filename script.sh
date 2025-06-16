# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/americano/rgb/2x --output data/NeRF-DS/americano --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/chickchicken/rgb/2x --output data/NeRF-DS/chickchicken --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/cut-lemon1/rgb/2x --output data/NeRF-DS/cut-lemon1 --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/espresso/rgb/2x --output data/NeRF-DS/espresso --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/hand1-dense-v2/rgb/2x --output data/NeRF-DS/hand1-dense-v2 --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/keyboard/rgb/2x --output data/NeRF-DS/keyboard --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/oven-mitts/rgb/2x --output data/NeRF-DS/oven-mitts --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/slice-banana/rgb/2x --output data/NeRF-DS/slice-banana --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/split-cookie/rgb/2x --output data/NeRF-DS/split-cookie --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2
# CUDA_VISIBLE_DEVICES=4 python extract_masks.py --img_path data/HyperNeRF/torchocolate/rgb/2x --output data/NeRF-DS/torchocolate --iou_th 0.88 --stability_score_th 0.95 --downsample_mask 2


#### Training Script ####

CUDA_VISIBLE_DEVICES=3 python train_4dgs.py --model_path "/node_data/urp25sp_daeen/Seg4DGaussian/out"
