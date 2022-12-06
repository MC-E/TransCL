import torch
import torch.nn as nn
from torch.nn import init
import cv2
import numpy as np
import random
import torch.nn.functional as F


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        #         print(input_.shape)
        grad_input = grad_output.clone()
        grad_input[input_ < -1] = 0
        grad_input[input_ > 1] = 0
        return grad_input


MyBinarize = MySign.apply


class CS_Sampling_bin(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32):
        super(CS_Sampling_bin, self).__init__()

        print('CS ratio: ', cs_ratio)

        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiG = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiB = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))

        self.Phi_scaleR = nn.Parameter(torch.Tensor([0.01]))
        self.Phi_scaleG = nn.Parameter(torch.Tensor([0.01]))
        self.Phi_scaleB = nn.Parameter(torch.Tensor([0.01]))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize

    def forward(self, x):
        Phi_R = MyBinarize(self.PhiR) * self.Phi_scaleR
        Phi_G = MyBinarize(self.PhiG) * self.Phi_scaleG
        Phi_B = MyBinarize(self.PhiB) * self.Phi_scaleB

        PhiWeight_R = Phi_R.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)

        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0, stride=self.blocksize, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)

        return x


class CS_Sampling(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32, im_size=384):
        super(CS_Sampling, self).__init__()
        print('bcs')

        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiG = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiB = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize

        self.im_size = im_size

    def forward(self, x):
        Phi_R = self.PhiR
        Phi_G = self.PhiG
        Phi_B = self.PhiB

        PhiWeight_R = Phi_R.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)

        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0, stride=self.blocksize, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)
        x = F.interpolate(x, size=(self.im_size, self.im_size), mode='bilinear')

        return x


class CS_Sampling_rm(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32, image_size=384, rate_rm=0.1, random=True):
        super(CS_Sampling_rm, self).__init__()

        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiG = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiB = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize

        self.random = random
        self.num_blocks = (image_size // blocksize) ** 2
        self.rate_rm = rate_rm
        self.image_size = image_size
        self.blocksize = blocksize

        num_x = image_size // blocksize
        self.pos = [[i, j] for j in range(num_x) for i in range(num_x)]

    def generate_mask(self, size, psize, num_rm):
        mask = torch.ones(size=(size, size))
        random.shuffle(self.pos)
        pos_rm = self.pos[:num_rm]
        for pos_rm_i in pos_rm:
            mask[pos_rm_i[0] * psize:(pos_rm_i[0] + 1) * psize, pos_rm_i[1] * psize:(pos_rm_i[1] + 1) * psize] = 0.
        return mask.unsqueeze(0).unsqueeze(0)

    def updata_rat(self, rat):
        self.rate_rm = rat

    def forward(self, x):
        #         print(self.rate_rm)
        img = x
        if self.random == True:
            rate_rm = torch.rand(1) * 0.5
        else:
            rate_rm = self.rate_rm

        num_rm = int(rate_rm * self.num_blocks)
        #         print(rate_rm,num_rm,self.num_blocks)
        masks = []
        for b in range(x.shape[0]):
            masks.append(self.generate_mask(size=self.image_size, psize=self.blocksize, num_rm=num_rm))
        masks = torch.cat(masks, dim=0).cuda()
        x = x * masks
        img = ((((img + 1.) / 2. * masks)[0].permute(1, 2, 0).cpu().data.numpy()) * 255.).astype(np.uint8)
        cv2.imwrite('mask.png', img)
        exit(0)

        Phi_R = self.PhiR
        Phi_G = self.PhiG
        Phi_B = self.PhiB

        PhiWeight_R = Phi_R.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)

        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0, stride=self.blocksize, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)

        return x


class CS_Sampling_shuffle(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32, image_size=384, rate_rm=0.1, random=True):
        super(CS_Sampling_shuffle, self).__init__()

        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiG = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiB = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize

        self.random = random
        self.num_blocks = (image_size // blocksize) ** 2
        self.rate_rm = rate_rm
        self.image_size = image_size
        self.blocksize = blocksize

        num_x = image_size // blocksize
        self.pos = [[i, j] for j in range(num_x) for i in range(num_x)]

    def shuffle(self, x):
        h, w = x.shape[-2], x.shape[-1]
        blocks = F.unfold(x, kernel_size=32, stride=32).permute(0, 2, 1)
        l = blocks.shape[1]
        idxes = list(range(l))
        random.shuffle(idxes)
        blocks = blocks[:, idxes, :]
        blocks = blocks.permute(0, 2, 1)
        return F.fold(blocks, output_size=(h, w), kernel_size=32, stride=32).contiguous()

    def forward(self, x):
        x = self.shuffle(x)
        Phi_R = self.PhiR
        Phi_G = self.PhiG
        Phi_B = self.PhiB

        PhiWeight_R = Phi_R.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)

        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0, stride=self.blocksize, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)

        return x


class CS_Sampling_arb(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32):
        super(CS_Sampling_arb, self).__init__()

        n_output = int(blocksize ** 2)

        self.PhiR = nn.Parameter(init.xavier_normal_(torch.Tensor(blocksize * blocksize, blocksize * blocksize)))
        self.PhiG = nn.Parameter(init.xavier_normal_(torch.Tensor(blocksize * blocksize, blocksize * blocksize)))
        self.PhiB = nn.Parameter(init.xavier_normal_(torch.Tensor(blocksize * blocksize, blocksize * blocksize)))

        self.n_channels = n_channels
        self.n_output = n_output
        self.blocksize = blocksize

    def forward(self, x, num_rows=None):
        if num_rows is None:
            num_rows = np.random.randint(1, 1024)

        Phi_R = self.PhiR[:num_rows, :]
        Phi_G = self.PhiG[:num_rows, :]
        Phi_B = self.PhiB[:num_rows, :]

        PhiWeight_R = Phi_R.contiguous().view(num_rows, 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(num_rows, 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(num_rows, 1, self.blocksize, self.blocksize)

        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0, stride=self.blocksize, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, num_rows, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, num_rows, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, num_rows, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)

        return x

if __name__ == '__main__':
    cs_sampling = CS_Sampling(n_channels=3, cs_ratio=0.25, blocksize=16, im_size=384)
    input_img = torch.randn(2, 3, 32, 32)
    output_img = cs_sampling(input_img)
    print(output_img.shape)