import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import  SequentialSampler, SubsetRandomSampler
import argparse
import os
from tqdm import tqdm
from random import shuffle
from load_data import pre_process_data_with_detector, Hand_Estimator_Dataloader
from models import ResNet18Enc_D, Decoder
from utils import weighted_mse_loss, calc_auc, EvalMetrics, feed_eval_util_3d

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch_size", type=int, default=32,
                    help="Training batch size")
parser.add_argument("-e","--epochs", type=int, default=11,
                    help="Number of training epochs")
parser.add_argument("-l","--lr", type=float, default=0.001, 
                    help="The learning rate to train the 3D-HPE")
parser.add_argument("-t","--train_test_switch", type=bool, default=True, 
                    help="TRUE to train the 3D-HPE, FALSE to test pre-trained model")
parser.add_argument('-m', "--model_dir", type=str, default='models/')
# parser.add_argument('-i', "--img_dir", type=str, default='/media/razvan/Work/ubuntu-backup/')
parser.add_argument('-i', "--img_dir", type=str, default='/media/razvan/db/')
parser.add_argument("-g","--gpus", type=int, nargs="+", default=[0],
                    help="GPUs to use")
# parser.add_argument('-s', "--set_name", type=str, default='icvl')
parser.add_argument('-s', "--set_name", type=str, default='mega')
args = parser.parse_args()

def test(models, testset, setname):
    models[0].eval()
    models[1].eval()
    print("Testing. \n")
    eval_metric = EvalMetrics()
    with torch.no_grad():
        for data in tqdm(testset, leave=False, total=len(testset)):
            images = data["image"].cuda()
            # target = data["keypoints"].cuda()
            keypoints = models[1](models[0](images))
            
            feed_eval_util_3d(eval_metric, data, keypoints, setname)
        mean, median, auc, pck_curve_all, threshs = eval_metric.get_measures(0.0021, 0.050, 6)
        pck_curve_all, threshs = pck_curve_all[8:], threshs[8:]
        auc_subset = calc_auc(threshs, pck_curve_all)
        print('Evaluation results:')
        print('Average mean EPE: %.3f mm' % (mean))
        print('Average median EPE: %.3f mm' % (median))
        print('Area under curve: %.3f' % auc)
        print('Area under curve between 20mm - 50mm: %.3f' % auc_subset)
        return (mean)
    
def run(args):
    # datasets = ["train", "test"]
    # pre_process_data_with_detector(args.model_dir, args.img_dir, datasets, 'mega')
    # # load Hand Detector
    # if os.path.isfile("%s/%s_unet_detector.pth"%(args.model_dir, 'mega')):
    #     ckpt = torch.load("%s/%s_unet_detector.pth"%(args.model_dir, 'mega'))
    #     detector_model.load_state_dict(ckpt["state_dict_detector"])
    #     detector_model = detector_model.cuda()
    #     detector_model.eval()
    
    # Load model

        
    encoder = nn.DataParallel(ResNet18Enc_D(128), device_ids=args.gpus).cuda()
    decoder = nn.DataParallel(Decoder(512), device_ids=args.gpus).cuda()

    if os.path.isfile('%s/%s_%s_estimator.pth' %(args.model_dir,  args.set_name, 
                                                   'resnet18_enc_regression')):
        ckpte = torch.load('%s/%s_%s_estimator.pth' %(args.model_dir,  args.set_name, 
                                                   'resnet18_enc_regression'))
        encoder.load_state_dict(ckpte["state_dict_encoder"])
        decoder.load_state_dict(ckpte["state_dict_decoder"])

    hand_config_file = {"anno_dir": args.img_dir,
                        "device": args.gpus, #["0"],
                        "set_name": args.set_name,
                        "image_dir": args.img_dir}
    data_loader = Hand_Estimator_Dataloader(hand_config_file, args.model_dir,  'train')
    training_idx = list(range(0, 200000))
    shuffle(training_idx)
    dataset = DataLoader(
                        data_loader,
                        batch_size=64,
                        # shuffle=True,
                        sampler=SubsetRandomSampler(training_idx[0:100000])
                        # sampler=SequentialSampler(range(25600)),
                        )
    test_loader = Hand_Estimator_Dataloader(hand_config_file, args.model_dir,  'test')
    testset = DataLoader(
                        test_loader,
                        batch_size=64,
                        # shuffle=False,
                        sampler=SequentialSampler(range(10000))
                        )
    if args.train_test_switch == 1:
        best_acc = 9999

        optim_backbone = optim.Adam(encoder.parameters(), lr=1e-3)
        optim_dec = optim.Adam(decoder.parameters(), lr=1e-3)  
        sched_backbone = lr_scheduler.ReduceLROnPlateau(optim_backbone, 
                                                        mode="min", 
                                                        factor=0.1, 
                                                        patience=2, 
                                                        threshold=0.01, 
                                                        cooldown=10, 
                                                        verbose=True)
        sched_dec = lr_scheduler.ReduceLROnPlateau(optim_dec, 
                                                    mode="min", 
                                                    factor=0.1, 
                                                    patience=2, 
                                                    threshold=0.01, 
                                                    cooldown=10, 
                                                    verbose=True)
        # criterion = torch.nn.MSELoss().cuda()
        # criterion = weighted_mse_loss().cuda()
        set_length = len(dataset)
        print("Training. \n")
        for epoch in range(args.epochs):
            encoder.train()
            decoder.train()
            total_loss_train = 0.0
            for x in tqdm(dataset, leave=False, total=set_length):
                images = x["image"].cuda()
                target = x["keypoints"].cuda()
                arg_loss = x["detected"].cuda()
                
                optim_backbone.zero_grad()
                optim_dec.zero_grad()
                
                latents = encoder(images)
                keypoints = decoder(latents)
                
                loss =  weighted_mse_loss(keypoints, target.reshape(-1, 63), arg_loss)
                loss.backward()
                
                optim_backbone.step()
                optim_dec.step()
                
                total_loss_train += loss.item()
                
            total_loss_train /= set_length
            print(total_loss_train)
            
            sched_backbone.step(total_loss_train)
            sched_dec.step(total_loss_train)
            
            if epoch % 5  == 0:
                acc = test([encoder,decoder], testset, args.set_name)
                if best_acc > acc:
                    best_acc = acc
                    torch.save({
                            'epoch': epoch + 1,
                            'state_dict_encoder': encoder.state_dict(),
                            'state_dict_decoder': decoder.state_dict(),
                        },
                        '%s/%s_%s_estimator2.pth' %(args.model_dir,  args.set_name, 
                                                   'resnet18_enc_regression'))
                print('Average precision: {:.3f} [mm] \t \
                    Best average precision: {:.3f} [mm]'.format(acc, best_acc))
    else:
        acc = test([encoder,decoder], testset, args.set_name)
        print('Average precision: {:.3f} [mm]'.format(acc))

if __name__ == '__main__':
    run(args)