import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from PIL import Image
from utils import xyz2uvd

class Hand_Detector_Dataloader(Dataset):
    """
    Dataloard for Hand Detection and processing
    """

    def __init__(self, config, set_type="train"):
        self.device = config["device"]
        self.image_dir = config["image_dir"]
        self.set_name = config['set_name']
        self.file_name, self.xyz_jnt_gt = [], []
        self.anno_dir = os.path.join(config["anno_dir"])


        hand_anno_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
        with open(self.anno_dir + set_type + "_annotation.txt", 'r', 
                    encoding='utf-8',newline='') as f:
            for line in f:
                part = line.split('\t')
                self.file_name.append(part[0])
                self.xyz_jnt_gt.append(part[1:64])
        self.xyz_jnt_gt = np.array(self.xyz_jnt_gt,dtype='float64')
        self.xyz_jnt_gt.shape = (self.xyz_jnt_gt.shape[0], 21, 3)
        self.xyz_jnt_gt = self.xyz_jnt_gt[:,hand_anno_index,:]
        self.uvd_jnt_gt = xyz2uvd(xyz=self.xyz_jnt_gt, setname=self.set_name)


    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        padWidth = 200
        image_name = self.file_name[idx]
        image = Image.open(os.path.join(self.image_dir, image_name + ".png"))
        image = np.asarray(image, dtype='uint16')
        image_orig = image / 2000.0
        if self.set_name == "mega":
            centerU = 315.94485
            centerJ = 9
        elif self.set_name == "icvl":
            centerU = 160
            centerJ = 0
        # self.transform = transforms.Compose([transforms.Resize(())
        #                                     transforms.ToTensor()])
        cur_uvd = self.uvd_jnt_gt[idx]

        bb = np.array([(50.0, 50.0, cur_uvd[centerJ, 2])])
        bbox_uvd = xyz2uvd(setname=self.set_name, xyz=bb)
        margin = int(np.ceil(bbox_uvd[0,0] - centerU))

        axis_bounds = np.array([np.min(cur_uvd[:, 0]), np.max(cur_uvd[:, 0]),
                                np.min(cur_uvd[:, 1]), np.max(cur_uvd[:, 1]),
                                np.min(cur_uvd[:, 2]), np.max(cur_uvd[:, 2])], dtype='int32')

        tmpDepth = np.zeros((image_orig.shape[0]+padWidth*2,image_orig.shape[1]+padWidth*2))
        tmpDepth[padWidth:padWidth+image_orig.shape[0],padWidth:padWidth+image_orig.shape[1]]=image

        crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
        loc = np.where(np.logical_and(crop>axis_bounds[4]-50.0,crop<axis_bounds[5]+50.0))
        cropmask=np.zeros_like(crop)

        cropmask[loc]=1
        orimask = np.zeros_like(tmpDepth, dtype='uint8')
        orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                        axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
        depth_mask = orimask[padWidth:padWidth+image_orig.shape[0],padWidth:padWidth+image_orig.shape[1]]
        image_orig = torch.tensor(image_orig)
        image_orig = torch.unsqueeze(image_orig, 0)
        depth_mask = torch.tensor(depth_mask)
        depth_mask = torch.unsqueeze(depth_mask, 0)
        return {
            "image": image_orig.float(),
            "depth_mask": depth_mask.float(),
        }
