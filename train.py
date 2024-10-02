import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, utils
from models.model import MEFNetwork, Fusion, init_parameters
from losses.mefssim import MEF_MSSSIM
from datasets.ImageDataset import ImageSeqDataset
from datasets.batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
EPS = 1e-8

class Train(object):
    def __init__(self, config):

        ############# trainting transforms ##############################
        torch.manual_seed(config.seed)
        self.train_hr_transform = transforms.Compose([
            BatchRandomResolution(config.high_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])
        
        self.train_lr_transform = transforms.Compose([
            BatchRandomResolution(config.low_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])
        self.test_lr_transform =  transforms.Compose([
           BatchRandomResolution(config.low_size, interpolation=2),
           BatchToTensor(),
           BatchRGBToYCbCr()
        ])
        self.test_hr_transform = transforms.Compose([
            BatchTestResolution(config.test_high_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])
        ############# training set configuration ##############################
        self.train_batch_size = 1
        self.test_batch_size = 1
        self.train_data = ImageSeqDataset(csv_file=os.path.join(config.trainset, 'train.txt'),
                                         hr_img_seq_dir=config.trainset,
                                         hr_transform=self.train_hr_transform,
                                         lr_transform=self.train_lr_transform)

        self.train_loader = DataLoader(self.train_data,
                                      batch_size=self.train_batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=4)
        
        self.test_data = ImageSeqDataset(csv_file=os.path.join(config.testset, 'test.txt'), # mfdb test set
                                         hr_img_seq_dir=config.testset,
                                         hr_transform=self.test_hr_transform,
                                         lr_transform=self.test_lr_transform)

        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=4)
        ############# train ##############################
        self.offline_test_size = config.offline_test_size
        self.luts_path = config.train_luts_path
        self.test_results = []
        self.train_loss = []
        self.n_frames = config.n_frames
        self.start_epoch = 0
        self.max_epochs = config.max_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                        last_epoch=self.start_epoch-1,
                                        step_size=config.decay_interval,
                                        gamma=config.decay_ratio)
        ############# initialize the model ##############################
        self.radius = config.radius
        self.eps = config.eps
        self.layers = config.layers
        self.width = config.width
        self.model = MEFNetwork(is_guided=True, n_frames=self.n_frames, radius=self.radius, eps=self.eps, layers=self.layers, width=self.width)
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
        self.ckpt_path = config.ckpt_path
        self.use_cuda = config.use_cuda
        self.fused_img_path = config.fused_img_path
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

    

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)

    
    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            # TODO: remove this after debugging
            i_hr, i_lr = sample_batched['I_hr'], sample_batched['I_lr']
            i_hr = torch.squeeze(i_hr, dim=0)
            i_lr = torch.squeeze(i_lr, dim=0)

            Y_hr = i_hr[:, 0, :, :].unsqueeze(1)
            Y_lr = i_lr[:, 0, :, :].unsqueeze(1)

            if step < self.start_step:
                continue

            I_hr = Variable(Y_hr)
            I_lr = Variable(Y_lr)
            # print("I_lr.shape:", I_hr.size())
            if self.use_cuda:
                I_hr = I_hr.cuda()
                I_lr = I_lr.cuda()

            self.optimizer.zero_grad()
            O_hr, _, _ = self.model(I_lr, I_hr)

            self.loss = -self.loss_fn(O_hr, I_hr) #+ self.loss_mask_tv(O_w)
            self.loss.backward()
            self.optimizer.step()
            q = -self.loss.data.item()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * q
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d) [MEF-SSIM = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()
            start_time = time.time()

        self.train_loss.append(loss_corrected)
        self.scheduler.step()
        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            self.generate_offline_lut()
            test_results = self.eval_1dluts(epoch)
            # test_results = self.eval(epoch)

            self.test_results.append(test_results)
            if test_results > self.max_eval_ssim:
                self.max_eval_epoch = epoch
                self.max_eval_ssim = test_results
                max_ssim_model_name = "best"
                model_name = '{}-{:0>5d}.pt'.format(max_ssim_model_name, epoch)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'test_results': self.test_results,
                    'best_results': self.max_eval_ssim
                }, model_name)
            print("project_name:", self.project_name)
            out_str = 'Epoch {} Testing: Average MEF-SSIM: {:.4f} max epoch {}, MEF-SSIM: {:.4f}'.format(epoch, test_results,
                                                                                               self.max_eval_epoch, self.max_eval_ssim)
            with open(self.ckpt_path + "/res_log.txt", 'a') as f:
                f.write(out_str + "\n")
            print(out_str)

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_results,
            }, model_name)

        return self.loss.data.item()
       
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

    
    def generate_offline_lut(self):
        weights = [open(os.path.join(self.luts_path, str(i) + "_weight.txt"), 'w') for i in range(self.n_frames)]
        for index in range(256):
            imgs = [np.ones([self.offline_test_size, self.offline_test_size, 3], np.uint8) * index for i in range(self.n_frames)]
            imgs = [Image.fromarray(img.astype(np.uint8)).convert('RGB') for img in imgs]

            I_hr = self.test_hr_transform(imgs)
            I_lr = self.test_lr_transform(imgs)

            i_hr = torch.stack(I_hr, 0).contiguous()
            i_lr = torch.stack(I_lr, 0).contiguous()

            i_hr = torch.squeeze(i_hr, dim=0)
            i_lr = torch.squeeze(i_lr, dim=0)

            Y_hr = i_hr[:, 0, :, :].unsqueeze(1)

            Y_lr = i_lr[:, 0, :, :].unsqueeze(1)

            I_hr = Variable(Y_hr)
            I_lr = Variable(Y_lr)

            if self.use_cuda:
                I_hr = I_hr.cuda()
                I_lr = I_lr.cuda()
            O_hr, W_hr, _ = self.model(I_lr, I_hr)
            [weights[i].write(str(np.mean(np.array([np.mean(W_hr[i:i+1].cpu().detach().numpy(), 0)]))) + ',') for i in range(self.n_frames)]

    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]
            t[t > 1] = 1
            t[t < 0] = 0
            utils.save_image(t, "%s/%s.jpg" % (path, name))
    
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

