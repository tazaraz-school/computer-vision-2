# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import kaleido
import plotly.io as pio
import plotly.graph_objects as go
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model import AsymmetricCroCo3DStereo

from tqdm import tqdm
import torch

import argparse 
from torch.serialization import add_safe_globals

add_safe_globals([argparse.Namespace])



from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Frame Extraction
def extract_multiple_frames(video_path, frame_indices, out_dir="frames",all_indices=False): 
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    count = 0
    paths = []

    base = video_path.split("/")[-1][:-len(".mp4")]

    while success:
        if count in frame_indices or all_indices:
            path = os.path.join(out_dir,f"{base}frame_{count}_.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            if len(paths) == len(frame_indices) and not all_indices:
                break
        success, frame = cap.read()
        count += 1
    cap.release()
    return paths

# Scene Reconstruction from Multiple Frames
def process_multiple_frames(frame_paths, model, device, batch_size, niter, schedule, lr):

    # Load frames into memory and resize
    images = load_images(frame_paths, size=512)

    # Create all possible image pairs
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    # Run inference to get predictions
    output = inference(pairs, model, device, batch_size=batch_size)

    # Align scene globally for 3D reconstruction
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    return scene

# Geometric Consistency Loss (Pairwise from Global Scene)
def compute_geometric_consistency_loss(scene, idx1, idx2, lambda_rgb=2.5, huber_delta=2.0):

    # Extract scene components
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # Build intrinsic matrix using focal length
    f = focals[idx1].detach().cpu().numpy().item()
    H, W = imgs[idx1].shape[:2]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

    # Extract and convert extrinsics to world coordinates
    pose1 = poses[idx1]
    pose2 = poses[idx2]
    T1 = np.linalg.inv(pose1.detach().cpu().numpy())
    T2 = np.linalg.inv(pose2.detach().cpu().numpy())
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Compute the fundamental matrix from the relative pose
    t_x = np.array([[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]])
    F = np.linalg.inv(K).T @ t_x @ R_rel @ np.linalg.inv(K)

    # Get corresponding 2D and 3D points (based on the confidence map on points)
    from dust3r.utils.geometry import xy_grid, find_reciprocal_matches
    pts2d_list, pts3d_list = [], []
    for i in [idx1, idx2]:
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    
    # Skip if no valid matches
    if pts3d_list[0].shape[0] == 0 or pts3d_list[1].shape[0] == 0:
        print(f"Skipping frame pair ({idx1}, {idx2}) due to lack of 3D points.")
        return 0.0, np.array([]), np.array([]), imgs, np.array([])

    # Find mutual nearest 3D matches between frames
    reciprocal_in_P2, nn2_in_P1, _ = find_reciprocal_matches(*pts3d_list)
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # Convert points to homogeneous coordinates for epipolar geometry
    matches_im0_h = np.concatenate([matches_im0, np.ones((matches_im0.shape[0], 1))], axis=1)
    matches_im1_h = np.concatenate([matches_im1, np.ones((matches_im1.shape[0], 1))], axis=1)

    # Compute epipolar lines in both images
    lines_in_im1 = (F @ matches_im0_h.T).T
    lines_in_im0 = (F.T @ matches_im1_h.T).T

    def point_line_dist(lines, points_h):
        num = np.abs(np.sum(lines * points_h, axis=1))
        denom = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2) + 1e-8
        return num / denom

    d1 = point_line_dist(lines_in_im1, matches_im1_h)
    d0 = point_line_dist(lines_in_im0, matches_im0_h)
    epipolar_distances = d0 + d1

    def huber(x, delta=huber_delta):
        return np.where(np.abs(x) < delta,
                        0.5 * x**2,
                        delta * (np.abs(x) - 0.5 * delta))

    rho = huber(epipolar_distances)

    # Sample corresponding RGB pixels
    def sample_rgb(img, coords):
        coords = np.clip(coords.round().astype(int), 0, np.array(img.shape[:2][::-1]) - 1)
        return img[coords[:, 1], coords[:, 0], :]

    rgb0 = sample_rgb(imgs[idx1], matches_im0)
    rgb1 = sample_rgb(imgs[idx2], matches_im1)
    rgb_l1 = np.abs(rgb0.astype(np.float32) - rgb1.astype(np.float32)).mean(axis=1)

    # Combine epipolar and RGB losses/metrics
    total_loss = np.mean(rho + lambda_rgb * rgb_l1)
    return total_loss, matches_im0, matches_im1, imgs, epipolar_distances

def save_epipolar_heatmap(matches_im0, distances, img,output_dir,input_path, vmax=20,):
    # Create heatmap
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    coords = np.clip(matches_im0.round().astype(int), 0, np.array(img.shape[:2][::-1]) - 1)
    for (x, y), d in zip(coords, distances):
        heatmap[y, x] = min(d, vmax)  # cap values at vmax

    # Prepare figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axs:
        ax.axis('off')

    # Left: Original image
    axs[0].imshow(img)

    # Right: Image with heatmap overlay
    axs[1].imshow(img)
    im = axs[1].imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=vmax)

    # Remove all margins/padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)


    os.makedirs(output_dir, exist_ok=True)
    base      = os.path.splitext(os.path.basename(input_path))[0]
    out_file  = os.path.join(output_dir, f"{base}heatmap.png")
    fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)                           


def all_frames_epipolar_analysis(video_path,model, device, batch_size_inference, batch_size_scene_construct,output_dir, niter, schedule, lr):
    frame_paths = extract_multiple_frames(video_path, [],all_indices=True) 
    print(f'extracted {len(frame_paths)} frames to apply.')

    range_loop = range(0, len(frame_paths), 1)
    scoring_list = []
    for scene_idx in tqdm(range_loop,desc="overall loop: constructing scene and saving epipolars"):
        # to do: make less scenes with dust3r
        if scene_idx + batch_size_scene_construct < len(frame_paths) : # skipping last |batch_size_scene_construct| amount of framesas nothing to compare to
            with suppress_stdout():
                frame_paths_scene = frame_paths[scene_idx:scene_idx + batch_size_scene_construct]
                scene = process_multiple_frames(frame_paths_scene, model, device, batch_size_inference, niter, schedule, lr)
                idx_pair = (0,0+ batch_size_scene_construct - 1) # isnt exclusive like prev code  so -1 
                score, matches_im0, _, imgs, distances = compute_geometric_consistency_loss(scene, *idx_pair) # np code into torch
                
                # curve showing score over time # all pairs, now will do consec

            if len(distances):
                save_epipolar_heatmap(matches_im0,distances,imgs[idx_pair[0]],output_dir=output_dir,input_path=frame_paths_scene[idx_pair[0]])
                scoring_list.append(score)
            else:
                print("no distances found at frame pair",idx_pair, " , still continuing.")
    if len(scoring_list) > 2:
        score_path = output_dir + "/scores/" + video_path.split("/")[-1].split(".")[0] + ".png"
        os.makedirs(os.path.dirname(score_path), exist_ok=True)
        plot_frame_values(scoring_list,output_path=score_path)
    

def frames_to_videos(frames_dir, fps=5, output_dir="videos_out"):
    """
    Combine JPEG frames into separate video files, grouped by video name.

    Args:
        frames_dir: Path to the directory containing JPEG frames
        fps: Frames per second
        output_dir: Where to save the output videos
    """
    frames_dir = Path(frames_dir)
    os.makedirs(output_dir, exist_ok=True)

    grouped_frames = defaultdict(list)

    for frame_path in frames_dir.glob("*.png"):
        filename = frame_path.name
        splitted = filename.split("_")
        frame_number = splitted[2]
        grouped_frames["_".join(splitted[0:2])].append((int(frame_number), frame_path))

    for video_name, frames in grouped_frames.items():
        sorted_frames = sorted(frames, key=lambda x: x[0])
        first_frame = cv2.imread(str(sorted_frames[0][1]))

        if first_frame is None:
            print(f"Error reading first frame for {video_name}")
            continue

        height, width, _ = first_frame.shape
        output_path = Path(output_dir) / f"{video_name}.mp4"

        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for _, frame_path in sorted_frames:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                out.write(frame)
            else:
                print(f" Could not read frame {frame_path}")

        out.release()
        print(f"Video created: {output_path}")

def plot_frame_values(values, output_path, title="Epipolar Consistency Error Over Frames"):
    """
    Plots and saves a line graph of values per frame as a PNG using Plotly.

    Parameters:
    - values: List[float or int] — one value per frame
    - output_path: str — full path to save PNG file (must end with .png)
    - title: str — plot title
    """
    if not output_path.endswith('.png'):
        raise ValueError("Output path must end with '.png'")

    frames = list(range(len(values)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames,
        y=values,
        mode='lines+markers',
        line=dict(shape='linear'),
        name='Value per Frame'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Frame Index",
        yaxis_title="Value",
        template='plotly_white',
        width=800,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig.write_image(output_path)  # Requires kaleido
    print(f" Scores plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epipolar analysis with Dust3r.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()
    model_name = "/home/scur2682/computer-vision-2/dust3r-main/naver/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    castle_fake_score = all_frames_epipolar_analysis(
        video_path=args.input_path,
        model=AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device),
        device=device,
        batch_size_inference=10, # batch size of dust3r itself, lower if local
        batch_size_scene_construct=10, # how many frames to construct dust3r
        output_dir ="/home/scur2682/computer-vision-2/dust3r-main/output_epipolars", # where stores epipolars
        niter=100,
        schedule="cosine",
        lr=0.001
    )
    
    #frames_to_videos(frames_dir="/home/scur2682/computer-vision-2/dust3r-main/output_epipolars",output_dir="/home/scur2682/computer-vision-2/dust3r-main/output_epipolars/vids")
    # run seperately when completely done

