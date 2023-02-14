import os
import cv2
import argparse
import torch
import torch.nn as nn
import numpy as np

from models import ResNet18Enc_D, Decoder
from utils import xyz2uvd, uvd2xyz, show_2d_hand_skeleton
from tqdm import tqdm
from PIL import Image
from load_data import Hand_Estimator_Dataloader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  SequentialSampler

def show_results(depth,uvd_pred):
        imgcopy=depth.copy()
        min = imgcopy.min()
        max = imgcopy.max()
        imgcopy = (imgcopy - min) / (max - min) * 255.
        imgcopy = imgcopy.astype('uint8')
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)

        imgcopy = show_2d_hand_skeleton(imgcopy=imgcopy, 
                                        uvd_pred=np.array(uvd_pred,dtype='int32'))
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', imgcopy)
        # cv2.waitKey(1)
        return imgcopy

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model_dir", type=str, default='models/')
# parser.add_argument('-i', "--img_dir", type=str, 
#                           default='/media/razvan/Work/ubuntu-backup/')
parser.add_argument('-i', "--img_dir", type=str, default='/media/razvan/db/')
parser.add_argument("-g","--gpus", type=int, nargs="+", default=[0],
                    help="GPUs to use")
# parser.add_argument('-s', "--set_name", type=str, default='icvl')
parser.add_argument('-s', "--set_name", type=str, default='mega')
args = parser.parse_args()

encoder = nn.DataParallel(ResNet18Enc_D(128), device_ids=args.gpus).cuda()
decoder = nn.DataParallel(Decoder(512), device_ids=args.gpus).cuda()


if os.path.isfile('%s/%s_%s_estimator.pth' %(args.model_dir,  args.set_name, 
                                                'resnet18_enc_regression')):
    ckpte = torch.load('%s/%s_%s_estimator2.pth' %(args.model_dir,  args.set_name, 
                                                'resnet18_enc_regression'))
    encoder.load_state_dict(ckpte["state_dict_encoder"])
    decoder.load_state_dict(ckpte["state_dict_decoder"])

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out1 = cv2.VideoWriter('bnn_hand.avi',fourcc,20.0,(640,480))

hand_config_file = {"anno_dir": args.img_dir,
                    "device": args.gpus, #["0"],
                    "set_name": args.set_name,
                    "image_dir": args.img_dir}
test_loader = Hand_Estimator_Dataloader(hand_config_file, args.model_dir,  'test')
testset = DataLoader(
                    test_loader,
                    batch_size=1,
                    sampler=SequentialSampler(range(400)),
                    shuffle=False,
                    )

with torch.no_grad():
    for data in tqdm(testset, leave=False, total=len(testset)):
        images = data["image"].cuda()
        keypoints = decoder(encoder(images))
        
        image = Image.open(os.path.join(args.img_dir, data["file_name"][0] + ".png"))
        image = np.asarray(image, dtype='uint16')
        
        margin = np.squeeze(data["margin"].detach().cpu().numpy())
        normuvd = np.squeeze(keypoints.detach().cpu().numpy())
        normuvd = normuvd.reshape(21,3)
        # XYZ conversion
        meanUVD = np.squeeze(data["mean_uvd"].detach().cpu().numpy(), axis=0)
        bbsize = 300.0
    
        if args.set_name =='icvl':
            centerU=320/2
        if args.set_name =='nyu':
            centerU=640/2
        if args.set_name =='msrc':
            centerU=512/2
        if args.set_name=='mega':
            centerU=315.944855
        # numImg=normuvd.shape[0]

        bbsize_array = np.ones((1,3))*bbsize
        bbsize_array[:,2]=meanUVD[0,2]
        bbox_uvd = xyz2uvd(setname=args.set_name, xyz=bbsize_array)
        normUVSize = np.array(np.ceil(bbox_uvd[:,0]) - centerU, dtype='int32')
        # normuvd=normuvd.reshape(21,3)
        uvd = np.empty_like(normuvd)
        U, V, D = meanUVD[:,0], meanUVD[:,1], meanUVD[:,2]
        uvd[:, 0]  = (normuvd[:, 0] * margin + U) 
        uvd[:, 1]  = (normuvd[:, 1] * margin + V) 
        uvd[:, 2]  = (normuvd[:, 2] * 300.0  + D) 
        
        uvd = xyz2uvd(setname=args.set_name, xyz=np.squeeze(data["xyz_gt"], axis=0))
        # uvd[:,2] = normuvd[:,2]*bbsize
        # uvd[:,0:2] = normuvd[:,0:2]*normUVSize.reshape(1,1)
        # uvd += np.squeeze(meanUVD)
        # xyz_pred = uvd2xyz(setname=args.set_name, uvd=uvd)
        out_img = show_results(image, uvd)
        out1.write(out_img)
    out1.release()
