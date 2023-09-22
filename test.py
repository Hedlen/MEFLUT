import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, utils
from models.model import MEFLUT Fusion, init_parameters
from losses.mefssim import MEF_MSSSIM
from datasets.ImageDataset import ImageSeqDataset
from datasets.batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
EPS = 1e-8

class Test(object):
    def __init__(self, config):

        ############# trainting and testing transforms ##############################
        torch.manual_seed(config.seed)
        self.test_hr_transform = transforms.Compose([
            BatchTestResolution(config.test_high_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])
        
        self.test_lr_transform = transforms.Compose([
            BatchTestResolution(config.low_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])

        ############# testing set configuration ##############################
        self.test_batch_size = 1
        self.test_data = ImageSeqDataset(csv_file=os.path.join(config.testset, 'test.txt'),
                                         hr_img_seq_dir=config.testset,
                                         hr_transform=self.test_hr_transform,
                                         lr_transform=self.test_lr_transform)

        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=4)

        ############# initialize the model ##############################
        self.radius = config.radius
        self.eps = config.eps

        self.layers = config.layers
        self.width = config.width

        self.model = MEFLUT(is_guided=True, n_frames=config.n_frames, radius=self.radius, eps=self.eps, layers=self.layers, width=self.width)
        init_parameters(self.model)
        self.model_name = type(self.model).__name__

        ############# loss ##############################
        self.loss_fn = MEF_MSSSIM(is_lum=True)
        self.config = config

        if torch.cuda.is_available() and config.use_cuda:
            self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()
        ##############GFU ##############################
        self.fusion = Fusion(is_guided=True, radius=self.radius, eps=self.eps)
        #############some parametes##############################
        self.test_results = []
        self.ckpt_path = config.ckpt_path
        self.use_cuda = config.use_cuda

        self.n_frames = config.n_frames
        self.luts_path = config.luts_path
        self.fused_img_path = config.fused_img_path

       
    def get_weight_map(self):
        num = self.n_frames
        fs = [open(os.path.join(self.luts_path, str(i) + "_weight.txt"), 'r') for i in range(num)]
        data = [f.read().split(',') for f in fs]
        self.weight_map = np.zeros([num, 256], np.double)
        for k in range(num):
            for j in range(256):
                d = float(data[k][j])
                if d > 1.0:
                    d = 1.0
                if d < 0.0:
                    d = 0.0
                self.weight_map[k][j] = d
        [f.close() for f in fs]

    def get_fusion_mask(self, small_imgs, weight_map, img_masks):
        for k in range(small_imgs.shape[0]):
            img_masks[k:k+1] = weight_map[k:k+1][:, small_imgs[k:k+1]]
        return img_masks.squeeze(0)

    def eval_1dluts(self, epoch):
        scores = []
        self.get_weight_map()
        for step, sample_batched in enumerate(self.test_loader, 0):
            i_hr, i_lr, case = sample_batched['I_hr'], sample_batched['I_lr'], sample_batched['case']
            i_hr = torch.squeeze(i_hr, dim=0)
            i_lr = torch.squeeze(i_lr, dim=0)
            Y_hr = i_hr[:, 0, :, :].unsqueeze(1)
            Cb_hr = i_hr[:, 1, :, :].unsqueeze(1)
            Cr_hr = i_hr[:, 2, :, :].unsqueeze(1)
            Wb = (torch.abs(Cb_hr - 0.5) + EPS) / torch.sum(torch.abs(Cb_hr - 0.5) + EPS, dim=0)
            Wr = (torch.abs(Cr_hr - 0.5) + EPS) / torch.sum(torch.abs(Cr_hr - 0.5) + EPS, dim=0)
            Cb_f = torch.sum(Wb * Cb_hr, dim=0, keepdim=True).clamp(0, 1).cuda()
            Cr_f = torch.sum(Wr * Cr_hr, dim=0, keepdim=True).clamp(0, 1).cuda()
            Y_lr = i_lr[:, 0, :, :].unsqueeze(1)
            I_hr = Variable(Y_hr)
            I_lr = Variable(Y_lr)
            if self.use_cuda:
                I_hr = I_hr.cuda()
                I_lr = I_lr.cuda()
            weight_map_n = torch.from_numpy(self.weight_map).cuda()
            small_imgs = (Y_lr * 255).long().cuda()
            img_masks = torch.zeros([small_imgs.shape[0], 1, small_imgs.shape[2], small_imgs.shape[3]],
                                    dtype=torch.float).cuda()
            fusion_mask = self.get_fusion_mask(small_imgs, weight_map_n, img_masks)
            fusion_mask = Variable(fusion_mask).cuda()
            O_hr, W_hr = self.fusion(I_lr, fusion_mask, I_hr)

            q = self.loss_fn(O_hr, I_hr).cpu()
            scores.append(q.data.numpy())

            O_hr_RGB = YCbCrToRGB()(torch.cat((O_hr, Cb_f, Cr_f), dim=1))
            self._save_image(O_hr_RGB, self.fused_img_path, str(case[0]).split('/')[-1])

        avg_quality = sum(scores) / len(scores)
        print("avg_quality:", avg_quality)
        return avg_quality

    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]
            t[t > 1] = 1
            t[t < 0] = 0
            utils.save_image(t, "%s/%s.jpg" % (path, name))

