import numpy as np
from dataset import *
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import glob
import pickle
import os

def get_shadow_object_paths(image_list):
    shadow_list = []
    object_list = []
    res_list = []
    for path in image_list:
        shadow_path_split = path.split("/")
        shadow_path_split[-4] += "_shadow"
        shadow_list.append(os.path.join("/", *shadow_path_split))

        object_path_split = path.split("/")
        object_path_split[-4] += "_object"
        object_list.append(os.path.join("/", *object_path_split))

        res_path_split = path.split("/")
        res_path_split[-4] += "_res"
        res_list.append(os.path.join("/", *res_path_split))
    return shadow_list, object_list, res_list

# TODO: Update list of images
image_paths = [
    "/../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/gen/216298.jpg",
    "/../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/gen/216389.jpg",
    "/../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/gen/216456.jpg",
    "/../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/real/216298.jpg",
    "/../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/real/216389.jpg",
    "/../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/real/216456.jpg"
]

shadow_paths = ["/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_object/test/gen/216298.jpg", 
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_object/test/gen/216389.jpg", 
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_object/test/gen/216456.jpg",
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_object/test/real/216298.jpg",
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_object/test/real/216389.jpg",
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_object/test/real/216456.jpg"]

object_paths = ["/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_shadow/test/gen/216298.jpg", 
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_shadow/test/gen/216389.jpg", 
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_shadow/test/gen/216456.jpg",
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_shadow/test/real/216298.jpg",
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_shadow/test/real/216389.jpg",
                "/../../../../../scratch-shared/scur0700/aidahailmary/Kandinsky_Indoor_OS/Kandinsky_Indoor_shadow/test/real/216456.jpg"]
# res_paths = get_shadow_object_paths(image_paths)

idx_to_class = { 0: 'real', 1: 'gen'}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(class_to_idx)
print(idx_to_class)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gradCam_ShadowObject():
    # ImageShadowObject:
    # TODO: update model path
    indoor_model = "./checkpoints/ShadowObject_indoor.pth"
    # outdoor_model = "./checkpoints/ShadowObject_outdoor.pth"
    model = torchvision.models.resnet50(weights = None)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(indoor_model))
    model.eval()
    model.to(device)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    test_set = ShadowObjectDataset(shadow_paths, object_paths, class_to_idx)
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=6)
    
    for i, (images, label) in enumerate(test_dataloader):
        images = images.to(device)
        predicted_class = idx_to_class[torch.max(model(images).data, 1)[1].item()] 
        print(f"{image_paths[i]} is {idx_to_class[label.item()]}, classified as {predicted_class}")
        # print(f"res_path: {res_paths[i]}")
        grayscale_cam = cam(input_tensor=images, targets=[ClassifierOutputTarget(1)])
        rgb_image = np.array(Image.open(image_paths[i])) / 255
        rgb_image = cv2.resize(rgb_image, (256, 256))[:,:,:3]
        visualization = show_cam_on_image(rgb_image, grayscale_cam[0,:], use_rgb=False)
        cv2.imwrite(f"gradCam/SO_{image_paths[i].split('/')[-1]}", visualization) #TODO Make sure gradCam/ folder exists
        print(f"saved to: ./gradCam/SO_{image_paths[i].split('/')[-1]}")
        print()

gradCam_ShadowObject()