import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import h5py
import numpy as np
import os
from utils import xyz2uvd, uvd2xyz
from models import UNet
from data_generator import Hand_Detector_Dataloader
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
from torchvision import transforms
from PIL import Image

def getLabelformMsrcFormat(base_path,dataset):

    xyz_jnt_gt=[]
    file_name = []
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    with open('%s/%s_annotation.txt'%(base_path,dataset), mode='r',encoding='utf-8',newline='') as f:
        for line in f:
            part = line.split('\t')
            # print(part)
            file_name.append(part[0])
            xyz_jnt_gt.append(part[1:64])
    f.close()

    xyz_jnt_gt = np.array(xyz_jnt_gt,dtype='float32')
    # print(xyz_jnt_gt.shape)

    xyz_jnt_gt.shape = (xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt = xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt = xyz2uvd(xyz=xyz_jnt_gt,setname='mega')

    return uvd_jnt_gt, xyz_jnt_gt, file_name

def pre_process_data_with_detector(model_dir, img_dir, datasets, setname):
    # Load Detector Model to pre-process the images from the dataset
    detector_model = UNet(1, 1).cuda()
    criterion = torch.nn.BCEWithLogitsLoss()
    hand_config_file = {"anno_dir": img_dir,
                        "device": ["0"],
                        "set_name": setname,
                        "image_dir": img_dir}
    test_loader = Hand_Detector_Dataloader(hand_config_file, "test")
    testset = DataLoader(
                        test_loader,
                        batch_size=16,
                        shuffle=False
                        )
    if os.path.isfile("%s/%s_unet_detector.pth"%(model_dir, setname)):
        ckpt = torch.load("%s/%s_unet_detector.pth"%(model_dir, setname))
        detector_model.load_state_dict(ckpt["state_dict_detector"])
        detector_model = detector_model.cuda()
        detector_model.eval()
        
        # Testing Loop
                
        # idx = 0
        # for data in tqdm(testset, leave=False, total=10):
        #     inputs = data['image'].cuda()
        #     # ref_mask = data['depth_mask'].cuda()
        #     rec_mask = detector_model(inputs) 
        #     rec_mask = torch.sigmoid(rec_mask)
        #     # loss = criterion(rec_mask, ref_mask)
        #     # print(loss.item())

        #     for img_o, img_r in zip(inputs, rec_mask):
        #         img_o = np.squeeze(img_o.detach().cpu().numpy())
        #         img_r = np.squeeze(img_r.detach().cpu().numpy())
        #         img_r = np.round(img_r)
        #         img_o = 255.0 * img_o
        #         img   = np.array([img_o, img_o, img_o])
        #         for i in range(img_r.shape[0]):
        #             for j in range(img_r.shape[1]):
        #                 if img_r[i, j] == 1.0:
        #                     img[:, i, j] = [170, 214, 160]
                
        #         img = np.transpose(img, (1,2,0))
        #         cv2.imwrite('data/img_%d.jpg'%idx, np.uint16(img))
        #     del inputs, rec_mask
        #     torch.cuda.empty_cache()
        #     idx += 1
        # exit()
    else:
        # Train U-Net detector 
        optimiser = optim.Adam(detector_model.parameters(), lr=1e-4)
        train_loader = Hand_Detector_Dataloader(hand_config_file, "train")
        trainset = DataLoader(train_loader,
                            batch_size=16,
                            shuffle=False)   


        for epoch in range(5):
            detector_model.train()
            # dev = torch.device("cuda:" + ','.join(str(c) for c in hand_config_file["device"]))
            for data in tqdm(trainset, leave=False, total=len(trainset)):

                inputs = data['image'].cuda()
                ref_mask = data['depth_mask'].cuda()

                optimiser.zero_grad()

                rec_mask = detector_model(inputs) 
                loss = criterion(rec_mask, ref_mask)

                loss.backward()
                optimiser.step()
            torch.save({
                        'epoch': epoch + 1,
                        'state_dict_detector': detector_model.state_dict()},
                        '%s/%s_unet_detector.pth' %(model_dir, setname))
    # Create a directory to save the pre-processed images

    # Alternatively create an h5py file with all the adjusted images and annotations
    for dataset in datasets:
        print(dataset)
        uvd, xyz_joint,file_name =getLabelformMsrcFormat(base_path=img_dir,dataset=dataset)

def load_data_h5(save_dir, dataset, joint_no):

    f = h5py.File('%s/%s.h5'%(save_dir, dataset), 'r')
    test_y= np.array([d for d in f['uvd_norm_gt'][...] if sum(sum(d))!=0]).reshape(-1,len(joint_no)*3)

    if dataset.startswith("test") or dataset.startswith("train"):
        test_x0 = np.array([d for d in f['img0'][...] if sum(sum(d))!=0])

        file_names = f['new_file_names'][...]
        meanUVD = f['uvd_hand_centre'][...]
    else:
        test_x0 = np.array([d for d in f['img'][...] if sum(sum(d))!=0])

        file_names = f['file_names'][...]
        meanUVD = f['uvd_hand_centre'][...]

    f.close()
    print(dataset,' loaded') #,test_x0.shape,test_y.shape)
    return np.expand_dims(test_x0,axis=-1), test_y, file_names, meanUVD


class Hand_Estimator_Dataloader(Dataset):
    """
    Dataloard for Hand Estimator and processing
    Runs the hand detector to center crop the hand for estimation
    """

    def __init__(self, config, model_dir, set_type="train"):
        self.device = config["device"]
        self.image_dir = config["image_dir"]
        self.set_name = config['set_name']
        self.file_name, self.xyz_jnt_gt = [], []
        self.anno_dir = os.path.join(config["anno_dir"])


        hand_anno_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
        if self.set_name == "mega":
            with open(self.anno_dir + set_type + "_annotation.txt", 'r', 
                        encoding='utf-8',newline='') as f:
                for line in f:
                    part = line.split('\t')
                    self.file_name.append(part[0])
                    self.xyz_jnt_gt.append(part[1:64])
        # elif self.set_name == "icvl":
        #     with open(self.anno_dir + set_type + "_annotation.txt", 'r', 
        #                 encoding='utf-8',newline='') as f:
        self.xyz_jnt_gt = np.array(self.xyz_jnt_gt,dtype='float64')
        self.xyz_jnt_gt.shape = (self.xyz_jnt_gt.shape[0], 21, 3)
        self.xyz_jnt_gt = self.xyz_jnt_gt[:,hand_anno_index,:]
        self.uvd_jnt_gt = xyz2uvd(xyz=self.xyz_jnt_gt, setname=self.set_name)
        
        self.detector_model = UNet(1, 1).cuda()
        if os.path.isfile("%s/%s_unet_detector.pth"%(model_dir, "mega")):
            ckpt = torch.load("%s/%s_unet_detector.pth"%(model_dir, "mega"))
            self.detector_model.load_state_dict(ckpt["state_dict_detector"])
            self.detector_model = self.detector_model.cuda()
            self.detector_model.eval()

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        padWidth = 100
        image_name = self.file_name[idx]
        image = Image.open(os.path.join(self.image_dir, image_name + ".png"))
        image = np.asarray(image, dtype='uint16')
        image_orig = image / 2000.0
        image_orig = torch.unsqueeze(torch.tensor(image_orig), 0).float()
        image_orig = torch.unsqueeze(image_orig, 0).cuda()
        mask = self.detector_model(image_orig)
        mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask)
        is_present = 0.0
        loc = np.where(mask>0.5)
        mean_uvd = np.array([(0.0, 0.0, 0.0)])
        if loc[0].shape[0]>30:
            is_present = 1.0
            
            depth_val = image[loc]
            U, V, D = np.mean(loc[1]), np.mean(loc[0]), np.mean(depth_val)
            if D < 10:
                is_present = 0.0
                margin = 0
                norm_hand = np.zeros((1, 128, 128), dtype=np.float32)
                norm_uvd = np.zeros((21, 3), dtype=np.float32)
            else:
                mean_uvd = np.array([(U, V, D)])
                mean_xyz = uvd2xyz(setname=self.set_name, uvd=mean_uvd)
                gt_mean = np.mean(self.xyz_jnt_gt[idx], axis=0)
                
                dist = np.sqrt(np.sum((gt_mean-mean_xyz)**2, axis=-1))
                if dist < (300.0 / 3): # hand size
                    bb = np.array([(300.0, 300.0, np.mean(depth_val))])
                    margin = int(np.ceil(xyz2uvd(xyz=bb,setname=self.set_name)[0, 0] - 315.94485))
                    tmpDepth = np.zeros((image.shape[0]+padWidth*2, 
                                        image.shape[1]+padWidth*2))
                    tmpDepth[padWidth:padWidth+image.shape[0],
                            padWidth:padWidth+image.shape[1]] = image
                    tdsz = tmpDepth.shape
                    if (U-margin/2+padWidth < 0)         or (V-margin/2+padWidth < 0) or \
                    (U+margin/2+padWidth > tdsz[1]-1) or (V+margin/2+padWidth > tdsz[0]-1):
                        is_present = 0.0
                        norm_hand = np.zeros((1, 128, 128), dtype=np.float32)
                        norm_uvd = np.zeros((21, 3), dtype=np.float32)
                    else:    
                        crop = tmpDepth[int(V-margin/2+padWidth):int(V+margin/2+padWidth),
                                        int(U-margin/2+padWidth):int(U+margin/2+padWidth)]
                        norm_hand = np.ones_like(crop)
                        loc_hand = np.where(crop>0)
                        norm_hand[loc_hand] = (crop[loc_hand] - D) / 300.0
                        
                        norm_hand = cv2.resize(norm_hand, (128, 128), interpolation=cv2.INTER_LINEAR)
                        norm_hand = np.float32(np.expand_dims(norm_hand, 0))
                        
                        norm_uvd  = self.uvd_jnt_gt[idx].copy()
                        norm_uvd[:, 0]  = (norm_uvd[:, 0] - U) / margin
                        norm_uvd[:, 1]  = (norm_uvd[:, 1] - V) / margin
                        norm_uvd[:, 2]  = (norm_uvd[:, 2] - D) / 300.0
                        norm_uvd = np.float32(norm_uvd)
                else:
                    norm_hand = np.zeros((1, 128, 128), dtype=np.float32)
                    norm_uvd = np.zeros((21, 3), dtype=np.float32)
                    margin   = 0
        else:
            norm_hand = np.zeros((1, 128, 128), dtype=np.float32)
            norm_uvd = np.zeros((21, 3), dtype=np.float32)
            margin   = 0
        return {"image":     norm_hand, 
                "keypoints": norm_uvd,
                "mean_uvd": mean_uvd,
                "xyz_gt": self.xyz_jnt_gt[idx], 
                "detected":  is_present,
                "file_name": self.file_name[idx],
                "margin": margin}