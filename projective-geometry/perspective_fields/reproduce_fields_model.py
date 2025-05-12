import torch
from torchvision.models import resnet50
import torch.nn as nn

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(os.path.dirname(current_dir)) + "/PerspectiveFields/"
sys.path.insert(0, package_root)
from perspective2d import perspectivefields_reproduce as perspectivefields  #################################################


class Fields_Extractor_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # add persepctive fields
        #version = 'PersNet-360Cities'
        self.pf_model = perspectivefields.PerspectiveFields().eval().cuda()
        self.pf_model.input_format = 'RGB'
        # changed size to 256 in config, make dynamic
        self.classifier = FieldsClassifier()

    def transform_maps(self, latitude_map, gravity_maps):
        #print("CHEEEEEK_NEEWWW\n",latitude_map)
        latitude_map = latitude_map / 90.0
        #joined_maps = torch.cat([latitude_map.unsqueeze(0), gravity_maps], dim = 0)
        joined_maps = torch.cat([latitude_map.unsqueeze(1), gravity_maps], dim = 1) # now have [batch,3,rest HW], 3 as lat is scalar per pixel and grav is 2d
        return joined_maps
    
    def forward(self, x,preprocess=False):
        # takes input cv2imread cuz perspective fields code want that sadly img_bgr = 
        if preprocess:
            images, x = self.pf_model.inference_batch(x)
        else:
            images = x.clone()
            x = self.pf_model.forward(x)

        reference = x[0]["pred_latitude_original"]
        lats = torch.stack([row['pred_latitude_original'] for row in x],dim=0).to(dtype=reference.dtype, device=reference.device) 
        gravs = torch.stack([row['pred_gravity_original'] for row in x],dim=0).to(dtype=reference.dtype, device=reference.device)
        x = self.transform_maps(lats,gravs)

        x = self.classifier.forward(x)
        if preprocess:
            return images,x
        else:
            return x
    
class FieldsClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(pretrained = False)

        nr_filters = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(nr_filters, 2)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    
    
def create_model(target_device):
    model = Fields_Extractor_Classifier()
    model.to(target_device)
    return model

def load_model(target_device, path_to_checkpoint): # edited in FieldsClassifier
    model = Fields_Extractor_Classifier()
    try:
        if target_device == "cpu":
            model.classifier.load_state_dict(torch.load(path_to_checkpoint, map_location=torch.device('cpu'))) 
        else:
            model.classifier.load_state_dict(torch.load(path_to_checkpoint))
        print("Successfully Loaded Saved Model")
    except Exception as error:
        print("Failed to load Saved Model")
        print(error)
    model.to(target_device)
    return model