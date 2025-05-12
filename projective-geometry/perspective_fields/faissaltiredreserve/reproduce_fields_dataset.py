import torch
from torch.utils.data import Dataset, DataLoader
from random import sample
import glob

class PerspectiveMapDataset(Dataset):
    def __init__(self, image_paths, class_to_idx):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.image_paths)
    
    def transform_maps(self, latitude_map, gravity_maps):
        latitude_map = latitude_map / 90.0
        joined_maps = torch.cat([latitude_map.unsqueeze(0), gravity_maps], dim = 0)
        return joined_maps
    
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]

        """"
        field_path = self.image_path_to_field_path(image_filepath) # edited 
        field = torch.load(field_path)
        
        latitude_map = field['pred_latitude_original']
        gravity_maps = field['pred_gravity_original']
        
        joined_maps = self.transform_maps(latitude_map,  gravity_maps)
        """
        label = image_filepath.split("/")[-2]
        label = self.class_to_idx[label]
        
        
        return image_filepath, label
    
    
def get_train_dataloaders(train_image_paths, val_image_paths, class_to_idx):
    train_dataset = PerspectiveMapDataset(train_image_paths, class_to_idx)
    val_dataset = PerspectiveMapDataset(val_image_paths, class_to_idx)

    train_dataloader = DataLoader(train_dataset, batch_size =512, shuffle = True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size = 512, shuffle = True, num_workers=6)

    return train_dataloader, val_dataloader


def get_test_dataloaders(test_image_paths, unconfident_image_paths, misclassified_image_paths, class_to_idx):  
    
    easy_image_paths = list(set(test_image_paths) - set(unconfident_image_paths + misclassified_image_paths))

    easy_image_paths = sample(easy_image_paths,15) #int(len(easy_image_paths)*0.1)) #eeeee

    # for debug98933
    easy_image_paths = ["/scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/real/98936.jpg","/scratch-shared/scur0700/Projective_Geometry_Fields/Kandinsky_Indoor/test/real/98933.jpg"]

    bs = 1 # 

    easy_dataset = PerspectiveMapDataset(easy_image_paths, class_to_idx)
    easy_dataloader = DataLoader(easy_dataset, batch_size=bs, shuffle=False) # True

    unconfident_dataset = PerspectiveMapDataset(unconfident_image_paths + misclassified_image_paths, class_to_idx)
    unconfident_dataloader = DataLoader(unconfident_dataset, batch_size=bs, shuffle=True)
    
    misclassified_dataset = PerspectiveMapDataset(misclassified_image_paths, class_to_idx)
    misclassified_dataloader = DataLoader(misclassified_dataset, batch_size=bs, shuffle=True)
    
    return easy_dataloader, unconfident_dataloader, misclassified_dataloader
