#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import math
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gsplat import rasterization
import numpy as np
from PIL import Image

TENSORBOARD_FOUND = False

# --- Minimal placeholder GaussianModel ---
class GaussianModel(torch.nn.Module):
    def __init__(self, xyz, scale, rotation, opacity, features, rgb):
        super().__init__()
        self.xyz = torch.nn.Parameter(xyz)
        self.scale = torch.nn.Parameter(scale)
        self.rotation = torch.nn.Parameter(rotation)
        self.opacity = torch.nn.Parameter(opacity)
        self.features = torch.nn.Parameter(features)
        self.rgb = torch.nn.Parameter(rgb)

# --- Minimal placeholder DeformModel ---
class DeformModel(torch.nn.Module):
    def __init__(self, input_dim=3, time_dim=1, hidden_dim=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim + time_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 10)  # 3 for d_xyz, 4 for d_rot, 3 for d_scale
        )
    def forward(self, xyz, t):
        x = torch.cat([xyz, t], dim=1)
        out = self.mlp(x)
        d_xyz = out[:, :3]
        d_rot = out[:, 3:7]
        d_scale = out[:, 7:10]
        return d_xyz, d_rot, d_scale

# --- Camera intrinsics/extrinsics helper ---
def get_camera_K_and_viewmat(viewpoint_camera):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    return K, viewmat

# --- Gsplat rendering with deformation ---
def render_with_gsplat(viewpoint_camera, gaussians, deform_model, time, bg_color, scaling_modifier=1.0, feature_dim=32):
    K, viewmat = get_camera_K_and_viewmat(viewpoint_camera)
    xyz = gaussians.xyz  # [N, 3]
    scale = gaussians.scale  # [N, 3]
    rotation = gaussians.rotation  # [N, 4]
    opacity = gaussians.opacity  # [N, 1]
    features = gaussians.features  # [N, feature_dim]
    rgb = gaussians.rgb  # [N, 3]
    d_xyz, d_rot, d_scale = deform_model(xyz, time.expand(xyz.shape[0], 1))
    xyz_deformed = xyz + d_xyz
    scale_deformed = scale + d_scale
    rotation_deformed = rotation + d_rot  # or quaternion multiplication
    nd_features = torch.cat([rgb, features], dim=1)
    render_colors, render_alphas, info = rasterization(
        means=xyz_deformed,
        quats=rotation_deformed,
        scales=scale_deformed * scaling_modifier,
        opacities=opacity.squeeze(-1),
        colors=nd_features,
        viewmats=viewmat[None],
        Ks=K[None],
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False
    )
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "info": info,
    }

# --- Checkpoint saving ---
def save_checkpoint(output_dir, gaussians, deform_model, camera_meta, step):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(gaussians.state_dict(), os.path.join(output_dir, f"gaussians_{step}.pth"))
    torch.save(deform_model.state_dict(), os.path.join(output_dir, f"deform_{step}.pth"))
    with open(os.path.join(output_dir, "scene_meta.json"), "w") as f:
        json.dump(camera_meta, f)

# --- Minimal placeholder Camera class ---
class DummyCamera:
    def __init__(self, image_width=128, image_height=128):
        self.FoVx = math.radians(60)
        self.FoVy = math.radians(45)
        self.image_width = image_width
        self.image_height = image_height
        self.world_view_transform = torch.eye(4, device="cuda")
        self.time = torch.tensor([0.0], device="cuda")
        self.original_image = torch.rand(3, image_height, image_width, device="cuda")

# --- Minimal load_data implementation ---
def load_data(data_root, downsample='2x'):
    # Load scene and dataset info
    with open(os.path.join(data_root, "scene.json")) as f:
        scene_json = json.load(f)
    with open(os.path.join(data_root, "dataset.json")) as f:
        dataset_json = json.load(f)

    train_ids = dataset_json["train_ids"]
    images = []
    cameras = []
    times = []
    camera_meta = {"scene": scene_json, "dataset": dataset_json}

    for idx, frame_id in enumerate(train_ids):
        # Camera
        cam_path = os.path.join(data_root, "camera", f"{frame_id}.json")
        with open(cam_path) as f:
            cam_json = json.load(f)
        cam = Camera(cam_json, scene_json)
        cameras.append(cam)

        # Image
        img_path = os.path.join(data_root, "rgb", downsample, f"{frame_id}.png")
        if not os.path.exists(img_path):
            img_path = img_path.replace('.png', '.jpg')
        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).cuda()  # [3, H, W]
        images.append(img)

        # Time (use idx or actual time if available)
        times.append(torch.tensor([float(idx)], device='cuda'))

    return images, cameras, times, camera_meta

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_rgb(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Set time for the current camera
        time = times[vind]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_with_gsplat(viewpoint_cam, gaussians, deform_model, time, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_with_gsplat, (gaussians, deform_model, time, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_checkpoint(scene.model_path, gaussians, deform_model, scene.camera_meta, iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

class Camera:
    def __init__(self, cam_json, scene_json):
        # Load orientation (3x3), position (3,), intrinsics, etc.
        self.orientation = np.array(cam_json["orientation"], dtype=np.float32)
        self.position = np.array(cam_json["position"], dtype=np.float32)
        self.focal_length = cam_json["focal_length"]
        self.principal_point = np.array(cam_json["principal_point"], dtype=np.float32)
        self.image_size = cam_json["image_size"]
        self.scene_scale = scene_json["scale"]
        self.center = np.array(scene_json["center"], dtype=np.float32)
        self.near = scene_json["near"]
        self.far = scene_json["far"]

        # Compute world_view_transform (4x4) as per your convention
        self.world_view_transform = self.compute_world_view_transform()
        self.FoVx, self.FoVy = self.compute_fov()
        self.image_width = self.image_size[0]
        self.image_height = self.image_size[1]

    def compute_world_view_transform(self):
        # 3x3 orientation, 3x1 position
        R = self.orientation
        t = self.position.reshape(3, 1)
        # 4x4 matrix
        W = np.eye(4, dtype=np.float32)
        W[:3, :3] = R
        W[:3, 3] = t[:, 0]
        return torch.from_numpy(W).cuda()

    def compute_fov(self):
        fx = self.focal_length
        fy = self.focal_length  # or use pixel_aspect_ratio if needed
        w, h = self.image_size
        FoVx = 2 * np.arctan(w / (2 * fx))
        FoVy = 2 * np.arctan(h / (2 * fy))
        return FoVx, FoVy

def initialize_gaussians_from_points(points_path, feature_dim=32):
    points = np.load(points_path)  # [N, 3]
    N = points.shape[0]
    xyz = torch.from_numpy(points).float().cuda()
    scale = torch.abs(torch.randn(N, 3, device='cuda')) * 0.01
    rotation = torch.randn(N, 4, device='cuda')
    opacity = torch.sigmoid(torch.randn(N, 1, device='cuda'))
    features = torch.randn(N, feature_dim, device='cuda')
    rgb = torch.sigmoid(torch.randn(N, 3, device='cuda'))
    return xyz, scale, rotation, opacity, features, rgb

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--feature_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_every', type=int, default=1000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)