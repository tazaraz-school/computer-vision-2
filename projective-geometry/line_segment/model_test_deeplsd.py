from lines_model import load_model
import torch
from tqdm import tqdm
import pickle
from lines_dataset import LineSegmentDataset, DataLoader
import glob
import os

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

save_path = "./checkpoints/Lines_combined.pt"
model = load_model(target_device=device_name, path_to_checkpoint=save_path)

with open("model_performance.csv", "w+") as f:
    f.write("video, real_score, generated_score, accuracy\n")

    for video in ["castle", "castle_gen", "garden", "garden_gen", "mountain", "mountain_gen", "river", "river_gen", "pathway", "pathway_gen"]:
        # if directory does not exist, continue
        if not os.path.exists(f"assets/videos/{video}.mp4_frames"):
            print(f"Directory {video} does not exist, skipping...")
            continue

        if not os.path.exists(f"./lines/{video}_lines.pkl"):
            print(f"Lines file {video} does not exist, skipping...")
            continue

        class_to_idx = {f'{video}.mp4_frames': 0}
        # Get all files in the {video} directory
        image_paths = sorted(glob.glob(f"assets/videos/{video}.mp4_frames/*"))
        image_path_to_lines = pickle.load(open(f"./lines/{video}_lines.pkl", "rb"))

        dataset = LineSegmentDataset(image_paths, image_path_to_lines, class_to_idx)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for images, labels in tqdm(dataloader, desc="testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # print(f"images: {labels}")
            # print(outputs.data)
            assigned_total_score = torch.sum(outputs.data, 0)
            _, per_image_prediction = torch.max(outputs.data, 1)
            accuracy = torch.sum(labels == per_image_prediction) / len(labels)
            print(f"real score: {assigned_total_score[0]}; generated score: {assigned_total_score[1]}")
            print(f"overall accuracy: {accuracy}")
            f.write(f"{video}, {assigned_total_score[0]}, {assigned_total_score[1]}, {accuracy}\n")
