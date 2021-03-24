import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.gramMatrix import StyleLoss
import torchvision
import torch.nn.functional as F


class GMMModel(BaseModel):
    def name(self):
        return 'GMMModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['GMM_1', 'GMM_2', 'GMM_3']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['image_mask', 'input_mask', 'warped_cloth']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['GMM']
        else:  # during test time, only load Gs
            self.model_names = ['GMM']

        # load/define networks
        self.netGMM = networks.define_GMM(self.opt, self.gpu_ids)
        self.netGMM = torch.nn.DataParallel(self.netGMM)
        use_sigmoid = opt.no_lsgan

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_GMM = torch.optim.Adam(self.netGMM.parameters(), lr=opt.lr, betas=(0.5, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_GMM)

    def set_input(self, input):
        self.real_image = input['base_image'].to(self.device)
        self.real_image_mask = input['base_image_mask'].to(self.device)
        self.real_cloth = input['base_cloth'].to(self.device)
        self.real_cloth_mask = input['base_cloth_mask'].to(self.device)
        self.input_cloth = input['input_cloth'].to(self.device)
        self.input_cloth_mask = input['input_cloth_mask'].to(self.device)

    def forward(self):
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.cloth_mask = self.real_cloth.mul(self.real_cloth_mask)
        self.input_mask = self.input_cloth.mul(self.input_cloth_mask)

        #cloth warping fake
        self.grid , self.theta = self.netGMM(self.real_image_mask, self.input_mask)
        self.warped_cloth = F.grid_sample(self.input_mask, self.grid, padding_mode='border')
        self.warped_mask = F.grid_sample(self.input_cloth_mask, self.grid, padding_mode='zeros')

        # cloth warping real
        real_grid, real_theta = self.netGMM(self.real_image_mask, self.cloth_mask)
        self.warped_cloth_real = F.grid_sample(self.image_mask, real_grid, padding_mode='border')
        self.warped_mask_real = F.grid_sample(self.real_cloth_mask, self.grid, padding_mode='zeros')

    def backward_GMM(self):
        self.loss_GMM_1 = self.criterionL1(self.warped_cloth_real, self.image_mask)
        self.loss_GMM_2 = self.criterionL1(self.warped_mask_real, self.real_image_mask)
        self.loss_GMM_3 = self.criterionL1(self.warped_mask, self.real_image_mask)
        self.loss_GMM = self.loss_GMM_1 + self.loss_GMM_2 + self.loss_GMM_3
        self.loss_GMM.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_GMM.zero_grad()
        self.backward_GMM()
        self.optimizer_GMM.step()
