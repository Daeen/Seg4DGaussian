import os
import sys
import requests
import numpy as np
import uuid
from tqdm import tqdm
from random import randint
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

# --- Mask-guided contrastive loss faithful to the paper ---
def mask_guided_contrastive_loss(rendered_feats, mask_vectors, T_p=0.75, T_n=0.5, K_p=100, K_n=100):
    """
    rendered_feats: (N, D) tensor of semantic features for sampled pixels
    mask_vectors: (N, M) binary mask membership for sampled pixels (M = num masks)
    """
    N = rendered_feats.shape[0]
    rendered_feats = F.normalize(rendered_feats, dim=1)
    C_F = torch.matmul(rendered_feats, rendered_feats.t())  # (N, N)
    C = (torch.matmul(mask_vectors, mask_vectors.t()) > 0).float()  # (N, N)
    iu = torch.triu_indices(N, N, offset=1)
    C_flat = C[iu[0], iu[1]]
    C_F_flat = C_F[iu[0], iu[1]]
    hard_pos_mask = (C_flat == 1) & (C_F_flat < T_p)
    hard_neg_mask = (C_flat == 0) & (C_F_flat > T_n)
    hard_pos_idx = torch.where(hard_pos_mask)[0]
    hard_neg_idx = torch.where(hard_neg_mask)[0]
    if hard_pos_idx.numel() > K_p:
        hard_pos_idx = hard_pos_idx[torch.randperm(hard_pos_idx.numel())[:K_p]]
    if hard_neg_idx.numel() > K_n:
        hard_neg_idx = hard_neg_idx[torch.randperm(hard_neg_idx.numel())[:K_n]]
    pos_loss = (C_F_flat[hard_pos_idx]).mean() if hard_pos_idx.numel() > 0 else 0.0
    neg_loss = (C_F_flat[hard_neg_idx]).mean() if hard_neg_idx.numel() > 0 else 0.0
    loss = neg_loss - pos_loss
    return loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    # --- Check for latest saved iteration ---
    latest_iter = -1
    for it in sorted(saving_iterations, reverse=True):
        point_cloud_path = os.path.join(dataset.model_path, f"point_cloud/iteration_{it}/point_cloud.ply")
        deform_path = os.path.join(dataset.model_path, f"deform/iteration_{it}/deform.pth")
        if os.path.exists(point_cloud_path) and os.path.exists(deform_path):
            latest_iter = it
            break
    if latest_iter > 0:
        print(f"Resuming training from iteration {latest_iter}.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Load scene and models ---
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)
    scene = Scene(dataset, gaussians, load_iteration=latest_iter if latest_iter > 0 else None)
    gaussians.training_setup(opt)
    if latest_iter > 0:
        deform.load_weights(dataset.model_path, iteration=latest_iter)
    # Always re-initialize max_radii2D after loading
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=gaussians.get_xyz.device)

    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    # --- Persistent mask sampling for Phase 3 ---
    image_name_to_mask_indices = {}
    mask_shape_printed = set()
    phase3_initialized = False
    progress_bar = tqdm(range(latest_iter + 1, opt.iterations + 1), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    for iteration in range(latest_iter + 1, opt.iterations + 1):
        iter_start.record()
        # Phase control: set which parameters are optimized
        if iteration <= 20000:
            # Phase 2: Joint training (Gaussian + deformation MLP)
            gaussians.change_optimization_target('GAUSSIAN')
            # Deformation MLP is updated as usual
        else:
            # Phase 3: Semantic feature learning (only semantic features, freeze geometry/MLP)
            gaussians.change_optimization_target('FEATURE')
            # --- Initialize persistent mask sampling for Phase 3 ---
            if not phase3_initialized:
                for cam in scene.getTrainCameras():
                    mask_path = os.path.join(dataset.source_path, 'masks', cam.image_name + '.pt')
                    masks = torch.load(mask_path)
                    if masks.dim() == 3 and masks.shape[0] < masks.shape[1]:
                        masks = masks.permute(1, 2, 0)
                    M = masks.shape[-1]
                    mask_indices = torch.randperm(M)[:getattr(opt, 'num_sampled_masks', 25)]
                    image_name_to_mask_indices[cam.image_name] = mask_indices
                    if cam.image_name not in mask_shape_printed:
                        print(f"Image {cam.image_name}: mask shape {masks.shape}, selected mask indices {mask_indices.tolist()}")
                        mask_shape_printed.add(cam.image_name)
                phase3_initialized = True

        

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
    
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        if iteration > 20000:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof,
                mask=None, norm_gaussian_features=True, is_smooth_gaussian_features=True, smooth_K=getattr(opt, 'smooth_K', 16)
            )
        else:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof,
                mask=None, norm_gaussian_features=True, is_smooth_gaussian_features=False, smooth_K=getattr(opt, 'smooth_K', 16)
            )
        image, viewspace_point_tensor, visibility_filter, radii, rendered_feats, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_gaussian_features"], render_pkg["depth"]
        
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
        # --- Phase 3: Semantic feature learning ---
        if iteration > 20000:
            # Load mask for current image
            mask_path = os.path.join(dataset.source_path, 'masks', viewpoint_cam.image_name + '.pt')
            if os.path.exists(mask_path):
                masks = torch.load(mask_path)  # (H, W, M) or (M, H, W)
                # check masks shape
                print(masks.shape)
                if masks.dim() == 3 and masks.shape[0] < masks.shape[1]:
                    masks = masks.permute(1, 2, 0)  # (H, W, M)
                H, W, M = masks.shape
                # Use persistent mask indices for this image
                mask_indices = image_name_to_mask_indices[viewpoint_cam.image_name]
                masks = masks[:, :, mask_indices]
                if viewpoint_cam.image_name not in mask_shape_printed:
                    print(f"Image {viewpoint_cam.image_name}: mask shape {masks.shape}, selected mask indices {mask_indices.tolist()}")
                    mask_shape_printed.add(viewpoint_cam.image_name)
                # Sample pixels (random each iteration)
                num_pixels = min(getattr(opt, 'num_sampled_pixels', 5000), H * W)
                flat_idx = torch.randperm(H * W)[:num_pixels]
                ys = flat_idx // W
                xs = flat_idx % W
                # Build mask vectors for sampled pixels
                mask_vectors = masks[ys, xs, :]  # (num_pixels, num_masks)
                mask_vectors = (mask_vectors > 0).float()
                # Extract semantic features for those pixels
                feats_hw = rendered_feats.permute(1, 2, 0)
                sampled_feats = feats_hw[ys, xs, :]  # (num_pixels, D)
                # Compute mask-guided contrastive loss
                sem_loss = mask_guided_contrastive_loss(
                    sampled_feats, mask_vectors,
                    T_p=getattr(opt, 'hard_positive_th', 0.75),
                    T_n=getattr(opt, 'hard_negative_th', 0.5),
                    K_p=100, K_n=100
                )
                loss = sem_loss

        loss.backward()
        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            # Defensive: ensure max_radii2D is correct shape and device
            if gaussians.max_radii2D.shape[0] != gaussians.get_xyz.shape[0]:
                gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=gaussians.get_xyz.device)
            visibility_filter = visibility_filter.to(gaussians.max_radii2D.device)
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            
            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter :
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
        # --- Optimizer step ---
        if iteration <= 20000:
            gaussians.optimizer["GAUSSIAN"].step()
            gaussians.optimizer["GAUSSIAN"].zero_grad(set_to_none=True)
            deform.optimizer.step()
            deform.optimizer.zero_grad()
        else:
            gaussians.optimizer["FEATURE"].step()
            gaussians.optimizer["FEATURE"].zero_grad(set_to_none=True)
    print("Training complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 6000, 7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 10_000,20_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    args, _ = parser.parse_known_args()
    args.iterations = 30_000
    args.warm_up = 3000
    args.densify_until_iter = 15_000
    ### after 20k iternation, we freeze the model's weights and begin  Gaussian feature learning for an additional 10k iterations
    ###  For each image, we randomly sample 25 masks (M′ = 25) generated by SAM and 5k pixels (Np = 5000), with K= 16 to get smooth Gaussian features
    ### DBSCAN clustering to get the final Gaussian features
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
