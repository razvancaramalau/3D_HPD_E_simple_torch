import numpy as np
import cv2
import torch
from matplotlib import pylab

def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * ((input - target) ** 2).mean())

def convert_depth_to_uvd(depth):
    v, u = pylab.meshgrid(range(0, depth.shape[0], 1), range(0, depth.shape[1], 1), indexing= 'ij')
    # print v[0,0:10]
    # print u[0,0:10]
    v = np.asarray(v, 'uint16')[:, :, np.newaxis]
    u = np.asarray(u, 'uint16')[:, :, np.newaxis]
    depth = depth[:, :, np.newaxis]
    uvd = np.concatenate((u, v, depth), axis=2)

    # print v.shape,u.shape,uvd.shape
    return uvd

class EvalMetrics:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred):
        """ Used to feed data to the class. 
        Stores the euclidean distance between gt and pred, when it is visible. """
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)
        keypoint_vis = np.squeeze(keypoint_vis) #.astype('bool')

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds

def feed_eval_util_3d(eval_util, in_data, out_data, setname):
        for idx in range(in_data["xyz_gt"].shape[0]):
            keypoint_xyz21, normuvd = (
                in_data["xyz_gt"][idx].view(21, 3),
                # in_data["keypoint_scale"][idx],
                out_data[idx].view(21, 3),
            )
            keypoint_xyz21 = np.squeeze(keypoint_xyz21.detach().cpu().numpy())
            # keypoint_scale = np.squeeze(keypoint_scale.detach().cpu().numpy())
            normuvd = np.squeeze(normuvd.detach().cpu().numpy())
            # XYZ conversion
            meanUVD = in_data["mean_uvd"][idx]
            bbsize = 300.0
        
            if setname =='icvl':
                centerU=320/2
            if setname =='nyu':
                centerU=640/2
            if setname =='msrc':
                centerU=512/2
            if setname=='mega':
                centerU=315.944855
            # numImg=normuvd.shape[0]

            bbsize_array = np.ones((1,3))*bbsize
            bbsize_array[:,2]=meanUVD[0,2]
            bbox_uvd = xyz2uvd(setname=setname,xyz=bbsize_array)
            normUVSize = np.array(np.ceil(bbox_uvd[:,0]) - centerU, dtype='int32')
            # normuvd=normuvd.reshape(21,3)
            uvd = np.empty_like(normuvd)
            uvd[:,2]=normuvd[:,2]*bbsize
            uvd[:,0:2]=normuvd[:,0:2]*normUVSize.reshape(1,1)
            uvd += np.squeeze(meanUVD.detach().cpu().numpy())

            xyz_pred = uvd2xyz(setname=setname,uvd=uvd)
        
        
            kp_vis = np.ones_like(keypoint_xyz21[:, 0])
            eval_util.feed(keypoint_xyz21, kp_vis, xyz_pred)

def calc_auc(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)

    return integral / norm

def xyz2uvd(setname,xyz):
    if setname=='mega':
        focal_length_x = 475.065948
        focal_length_y = 475.065857
        u0= 315.944855
        v0= 245.287079

        uvd = np.empty_like(xyz)
        if len(uvd.shape)==3:
            trans_x= xyz[:,:,0]
            trans_y= xyz[:,:,1]
            trans_z = xyz[:,:,2]
            uvd[:,:,0] = u0 +focal_length_x * ( trans_x / trans_z )
            uvd[:,:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
            uvd[:,:,2] = trans_z #convert m to mm
        else:
            trans_x= xyz[:,0]
            trans_y= xyz[:,1]
            trans_z = xyz[:,2]
            uvd[:,0] = u0 +  focal_length_x * ( trans_x / trans_z )
            uvd[:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
            uvd[:,2] = trans_z #convert m to mm
        return uvd

    if setname =='msrc' or setname =='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor

    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240
        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
    if setname =='nyu'or setname =='NYU':
        res_x = 640
        res_y = 480
        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor

    uvd = np.empty_like(xyz)
    if len(xyz.shape)==3:
        trans_x= xyz[:,:,0]
        trans_y= xyz[:,:,1]
        trans_z = xyz[:,:,2]
        uvd[:,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
        uvd[:,:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
        uvd[:,:,2] = trans_z #convert m to mm
    else:
        trans_x= xyz[:,0]
        trans_y= xyz[:,1]
        trans_z = xyz[:,2]
        uvd[:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
        uvd[:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
        uvd[:,2] = trans_z #convert m to mm
    return uvd

def uvd2xyz(setname,uvd):
    if setname =='mega':
        focal_length_x = 475.065948
        focal_length_y = 475.065857
        u0= 315.944855
        v0= 245.287079
        xyz = np.empty_like(uvd)
        if len(uvd.shape)==3:

            xyz[:,:,2]=uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - u0)/focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- v0)/focal_length_y*xyz[:,:,2]
        else:

            z =  uvd[:,2] # convert mm to m
            xyz[:,2]=z
            xyz[:,0] = ( uvd[:,0]- u0)/focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- v0)/focal_length_y*z
        return xyz

    if setname =='msrc' or setname=='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor

    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
    if setname =='nyu' or setname=='NYU':
        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor
    # focal_length = np.sqrt(focal_length_x ^ 2 + focal_length_y ^ 2);
    if len(uvd.shape)==3:
        xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
        xyz[:,:,2]=uvd[:,:,2]
        xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
        xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
    else:
        xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
        z =  uvd[:,2] # convert mm to m
        xyz[:,2]=z
        xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
        xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z

    return xyz

def xyz2uvd_nyu(setname, xyz, jnt_type=None):

    if setname =='nyu'or setname =='NYU':
        res_x = 640
        res_y = 480
        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor
    else:
        exit('wrong convert for the dataset')

    uvd = np.empty_like(xyz)
    # if jnt_type != 'single':
    #     trans_x= xyz[:,:,0]
    #     trans_y= xyz[:,:,1]
    #     trans_z = xyz[:,:,2]
    #     uvd[:,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
    #     uvd[:,:,1] = res_y / 2 - res_y * focal_length_y * ( trans_y / trans_z )
    #     uvd[:,:,2] = trans_z #convert m to mm
    # else:
    trans_x= xyz[:,0]
    trans_y= xyz[:,1]
    trans_z = xyz[:,2]
    uvd[:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
    uvd[:,1] = res_y / 2 - res_y * focal_length_y * ( trans_y / trans_z )
    uvd[:,2] = trans_z #convert m to mm
    return uvd
    
def uvd2xyz_nyu(setname,uvd, jnt_type=None):

    if setname =='nyu' or setname=='NYU':
        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor
    else:
        exit('wrong convert for the dataset')

    # focal_length = np.sqrt(focal_length_x ^ 2 + focal_length_y ^ 2);
    if jnt_type == None:
        xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
        xyz[:,:,2]=uvd[:,:,2]
        xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
        xyz[:,:,1] = ( -uvd[:,:,1]+ res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
    else:
        xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
        z =  uvd[:,2] # convert mm to m
        xyz[:,2]=z
        xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
        xyz[:,1] = ( -uvd[:,1]+ res_y / 2)/res_y/focal_length_y*z

    if setname =='msrc' or setname=='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
        if len(uvd.shape)==3:
            xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
            xyz[:,:,2]=uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
        else:
            xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
            z =  uvd[:,2] # convert mm to m
            xyz[:,2]=z
            xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z
    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
        if len(uvd.shape)==3:
            xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
            xyz[:,:,2]=uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
        else:
            xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
            z =  uvd[:,2] # convert mm to m
            xyz[:,2]=z
            xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z
    if setname =='nyu' or setname=='NYU':
        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor
        #focal_length_x = 1.08836710
        #focal_length_y = 0.817612648
    # focal_length = np.sqrt(focal_length_x ^ 2 + focal_length_y ^ 2);
        if len(uvd.shape)==3:
            xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
            xyz[:,:,2]=uvd[:,:,2]
            # norm_x = uvd[:,:,0] / res_x - 0.5
            # norm_y = 0.5 - uvd[:,:,1] / res_y
            # xyz[:,:,0] = norm_x * focal_length_x * xyz[:,:,2]
            # xyz[:,:,1] = norm_y * focal_length_y * xyz[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
        else:
            xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
            xyz[:,2] = uvd[:,2]
            # norm_x = uvd[:,0] / res_x - 0.5
            # norm_y = 0.5 - uvd[:,1] / res_y
            # xyz[:,0] = norm_x * focal_length_x * xyz[:,2]
            # xyz[:,1] = norm_y * focal_length_y * xyz[:,2]
            xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z

    return xyz

colors = np.array([[0.,0,0],
              [1.0,.0,0],
              [0.8,.0,0],
              [0.6,0,0],
              [0.4,0,0],

              [0,1,0],
              [0,0.8,0],
              [0,0.6,0],
              [0,0.4,0],

              [0,0,1],
              [0,0,0.8],
              [0,0,0.6],
              [0,0,0.4],

              [1,1,0],
              [1,0.8,0],
              [1.,0.6,0],
              [1,0.4,0],

              [1,0,1],
              [1.,0,0.8],
              [1,0,0.6],
              [1,0,0.4],
              ]).reshape(21,3)*255

def show_2d_hand_skeleton(imgcopy, uvd_pred):

    ratio_size=int(1500.0/np.mean(uvd_pred[:,2]))
    for k in [1,5,9,13,17]:

        cv2.line(imgcopy, tuple(uvd_pred[0,0:2]), 
                          tuple(uvd_pred[k,0:2]), 
                          tuple(colors[k]), ratio_size)
        cv2.line(imgcopy, tuple(uvd_pred[k,0:2]), 
                          tuple(uvd_pred[k+1,0:2]), 
                          tuple(colors[k+1]), ratio_size)
        cv2.line(imgcopy, tuple(uvd_pred[k+1,0:2]), 
                          tuple(uvd_pred[k+2,0:2]), 
                          tuple(colors[k+2]), ratio_size)
        cv2.line(imgcopy, tuple(uvd_pred[k+2,0:2]), 
                          tuple(uvd_pred[k+3,0:2]), 
                          tuple(colors[k+3]), ratio_size)
    ratio_size=int(3000.0/np.mean(uvd_pred[:,2]))
    for j in range(uvd_pred.shape[1]):
        cv2.circle(imgcopy, (int(uvd_pred[j,0]),
                             int(uvd_pred[j,1])), ratio_size, 
                             tuple(colors[j]), -1)
    return imgcopy
