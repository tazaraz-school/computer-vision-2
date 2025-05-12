ROC = True
GRADCAM = False



import os
import numpy as np
import glob
from pandas.core.common import flatten

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import argparse


from reproduce_fields_dataset import get_test_dataloaders
from reproduce_fields_model import load_model
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_all_paths(base_path, data_paths):
    all_train_paths = []
    all_val_paths = []
    all_test_paths = []

    for data_path in data_paths:
        data_path = base_path + data_path

        train_paths = []
        val_paths = []
        test_paths = []
        
        for path in glob.glob(data_path):
            train_paths.append(glob.glob(path + "/train/*/*"))
            val_paths.append(glob.glob(path + "/val/*/*"))
            test_paths.append(glob.glob(path + "/test/*/*"))
            

        train_paths = list(flatten(train_paths)) 
        val_paths = list(flatten(val_paths))
        test_paths = list(flatten(test_paths))        

        all_train_paths += train_paths
        all_val_paths += val_paths
        all_test_paths += test_paths

    all_train_paths = list(set(all_train_paths))
    all_val_paths = list(set(all_val_paths))
    all_test_paths = list(set(all_test_paths))

    return all_train_paths, all_val_paths, all_test_paths

def test(model, test_dataloader, save_path, test_type):

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_pred_probs = torch.tensor([]).to(device)

    for images, labels in tqdm(test_dataloader, desc="testing"):
        with torch.no_grad(): # edit: put it inside loop to allow cam grad to use grad

            #images = images.to(device)
            #labels = labels.to(device)

            labels = labels.to(device)
            paths = images # scuffed ik but need og paths for gradcam

            images,outputs = model.forward([cv2.imread(imagepath) for imagepath in images],preprocess=True) # edited (was just images as input)  + "images" output for gradcam

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted = torch.cat((all_predicted, predicted))
            all_labels = torch.cat((all_labels, labels))
            all_pred_probs = torch.cat((all_pred_probs, outputs.data))

        if GRADCAM:
            target_layers = [model.classifier.resnet.layer4[-1]]  # Construct the CAM object once, and then re-use it on many images.
            with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                targets = [ClassifierOutputTarget(int(class_idx)) for class_idx in labels]

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=images, targets=targets,)
                cam_map = grayscale_cam[0] 

                # === Prepare original image (normalized tensor → numpy RGB image in range [0, 1]) ===
                from PIL import Image
                import torchvision.transforms as transforms
                image = Image.open(paths[0])

                # Define a transform to convert PIL 
                # image to a Torch tensor
                transform = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float)
                ])

                # transform = transforms.PILToTensor()
                # Convert the PIL image to Torch tensor
                img = transform(image).detach().cpu().numpy().transpose(1, 2, 0)
                print("eeeeeeeeeee",img.shape)
                #image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                #image_np = np.clip(, 0, 1)
                

                # === Create heatmap overlay ===
                cam_overlay = show_cam_on_image(img, cam_map, use_rgb=True)  # shape: [H, W, 3], uint8

                # === Combine original and heatmap side by side ===
                original_image_uint8 = (img * 255).astype(np.uint8)
                side_by_side = np.hstack((original_image_uint8, cam_overlay))  # [H, W*2, 3]

                # === Save the image ===
                save_dir = os.path.abspath("./plots/heatmap")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "cam_comparison.jpg")
                cv2.imwrite(save_path, side_by_side[:, :, ::-1])  # Convert RGB to BGR for OpenCV saving
                print(f"Saved CAM comparison to: {save_path}")

    
    if ROC:
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(all_labels.cpu(), all_pred_probs[:,1].cpu())
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plot_name = save_path.split("/")[-1].split(".")[-2]
        plt.title(f'ROC: {plot_name}_{test_type}')
        plt.legend(loc="lower right")
        plt.savefig(f"./plots/{plot_name}_{test_type}.png")
        with open(f"./plots/{plot_name}_{test_type}.pkl", 'wb') as f:
            pickle.dump([fpr, tpr, roc_auc], f)

        conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
        print(f"{plot_name} {test_type}")
        print("ROC curve area:", roc_auc)
        print(conf_matrix)
        print(f"{conf_matrix[0].sum().item()} real images, {conf_matrix[1].sum().item()} generated images")
        tn = conf_matrix[0,0]
        tp = conf_matrix[1,1]
        fp = conf_matrix[0,1]
        fn = conf_matrix[1,0]
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
        accuracy = 100 * correct / total
        print(f"Accuracy for {save_path}: {accuracy}")
        print()

    
    


if __name__ == "__main__":

    idx_to_class = {0: 'real', 1: 'gen'}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    print(class_to_idx)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default="indoor", help='choose: indoor, outdoor or combined')
    args = parser.parse_args()

    category = args.category
    print("category:", category)

    #base_path = "../dataset/"
    base_path = "../../../../../../scratch-shared/scur0700/Projective_Geometry_Fields/" # edited for reproduce, cant store local

    misclassified_indoor_file = "./misclassified_indoor_list.pkl"
    misclassified_outdoor_file = "./misclassified_outdoor_list.pkl"
    misclassified_combined_file = "./misclassified_combined_list.pkl"

    unconfident_indoor_file = "./unconfident_indoor_list.pkl"
    unconfident_outdoor_file = "./unconfident_outdoor_list.pkl"
    unconfident_combined_file = "./unconfident_combined_list.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if category == "indoor":
        image_data_paths = ["Kandinsky_Indoor"]

        misclassified_image_paths = sorted(pickle.load(open(misclassified_indoor_file, "rb")))
        unconfident_image_paths = sorted(pickle.load(open(unconfident_indoor_file, "rb")))

        for type_path in [misclassified_image_paths,unconfident_image_paths]: # edit 
            for path_idx in range(len(type_path)):
                type_path[path_idx] = base_path + type_path[path_idx][len('../dataset/'):]
        
        save_path = "./checkpoints/Fields_indoor.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    elif category == "outdoor":

        image_data_paths = ["Kandinsky_Outdoor"]

        misclassified_image_paths = sorted(pickle.load(open(misclassified_outdoor_file, "rb")))
        unconfident_image_paths = sorted(pickle.load(open(unconfident_outdoor_file, "rb")))
        
        save_path = "./checkpoints/Fields_outdoor.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    elif category == "combined":
        image_data_paths = ["Kandinsky_Indoor", "Kandinsky_Outdoor"]
        
        misclassified_image_paths = sorted(pickle.load(open(misclassified_combined_file, "rb")))
        unconfident_image_paths = sorted(pickle.load(open(unconfident_combined_file, "rb")))
        
        save_path = "./checkpoints/Fields_combined.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    train_image_paths, val_image_paths, test_image_paths = load_all_paths(base_path, image_data_paths) 
    
    easy_dataloader, unconfident_dataloader, misclassified_dataloader = get_test_dataloaders(test_image_paths, unconfident_image_paths, misclassified_image_paths, class_to_idx)

    test(model, easy_dataloader, save_path, "easy")
    #print("Testing on unconfident set with length of ", len(unconfident_dataloader)) 
    #test(model, unconfident_dataloader, save_path, "unconfident")
    #print("Testing on misclassified set with length of ", len(misclassified_dataloader))
    #test(model, misclassified_dataloader, save_path, "misclassified")
    