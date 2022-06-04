""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from spconv_utils import replace_feature, spconv
from functools import partial


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, planes2, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, padding= 1, stride=stride, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes2, kernel_size=3, padding= 1, stride=stride, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes2)
        self.stride = stride

        self.bn0 = norm_fn(inplanes)

    def forward(self, x, bnf):
        if bnf :
            x = replace_feature(x, self.bn0(x.features))

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        out = replace_feature(out, self.relu(out.features))

        return out


class STWDNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.pfe = nn.Sequential(
            nn.Conv1d(14, 64, kernel_size=1 ),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, embed_dim, kernel_size=1 ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1 ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1 ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
        )
        self.sfe = nn.Sequential(
            nn.Conv1d(15, 64, kernel_size=1 ),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, embed_dim, kernel_size=1 ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1 ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1 ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
        )

        self.ofe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),

        )
        self.cfe = nn.Sequential(
            nn.Linear(4, embed_dim)
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 8, stride = 4)
        self.avgpool1 = nn.AvgPool2d(kernel_size = 8, stride = 4)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        self.bnraw1 = nn.BatchNorm1d(14)
        self.bnraw2 = nn.BatchNorm1d(15)
        self.bnraw3 = nn.BatchNorm2d(11)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.rfe1 = SparseBasicBlock(11, 64, 256, norm_fn=norm_fn, indice_key='res1')
        self.rfe2 = SparseBasicBlock(256, 256, 256, norm_fn=norm_fn, indice_key='res2')


    # size of the inputs of forward [200, 14], [200, 39, 5, 14], [200, 39, 5, 1], [200, 39, 5, 1], [200, 16, 16, 11], [200, 16, 16, 1]
    def forward(self, pointfeature, stationfeature, stationpm, stationmask, regionfeature, regionmask):
        pointfeature = pointfeature.unsqueeze(-1)

        regionfeature = regionfeature.permute(0, 3, 1, 2)
        regionmask = regionmask.permute(0, 3, 1, 2)

        stationfeature = stationfeature.reshape(200, -1, 14).transpose(1,2)
        stationpm = stationpm.reshape(200, -1, 1).transpose(1,2)
        stationmask = stationmask.reshape(200, -1, 1).transpose(1,2)

        pointfeature = self.bnraw1(pointfeature)

        pf1 = self.pfe(pointfeature)

        dstf = stationfeature - pointfeature
        dstf = torch.cat([dstf, stationpm], 1)

        dstf = self.bnraw2(dstf)
        dstf = self.sfe(dstf)
        dstf = dstf * stationmask
        dstf = torch.sum(dstf, 2)
        ssm = torch.sum(stationmask, 2)
        dstf = dstf/(ssm + 0.0000001)


        # with the help of the spconv lib now using the 3*3 conv is better than 1*1 conv
        regionfeature = regionfeature - regionfeature[:,:,8:9,8:9]
        dicuv = torch.where(regionmask.squeeze(1) == 1)
        fearegion = regionfeature[dicuv[0], :, dicuv[1], dicuv[2]]
        dicuv = torch.cat([dicuv[0].unsqueeze(1),dicuv[1].unsqueeze(1),dicuv[2].unsqueeze(1)], 1)
        regionfeaturesp = spconv.SparseConvTensor(
            features=fearegion,
            indices=dicuv.int(),
            spatial_shape=[16,16],
            batch_size=200
        )
        rf1 = self.rfe1(regionfeaturesp, True)
        rf1 = rf1.dense()

        rf1 = rf1 * regionmask
        avgrf1 = self.avgpool1(rf1)
        avgrmask1 = self.avgpool1(regionmask)
        maxmask = self.maxpool(regionmask)
        avgrf1 = avgrf1/(avgrmask1 + 0.0000001)
        avgrf1 = avgrf1 * maxmask

        dicuv2 = torch.where(maxmask.squeeze(1) == 1)
        fearegion2 = avgrf1[dicuv2[0], :, dicuv2[1], dicuv2[2]]
        dicuv2 = torch.cat([dicuv2[0].unsqueeze(1), dicuv2[1].unsqueeze(1), dicuv2[2].unsqueeze(1)], 1)
        regionfeaturesp2 = spconv.SparseConvTensor(
            features=fearegion2,
            indices=dicuv2.int(),
            spatial_shape=[3,3],
            batch_size=200
        )
        rf2 = self.rfe2(regionfeaturesp2, False)
        rf2 = rf2.dense()


        rf2 = rf2 * maxmask
        avgrf2 = self.avgpool2(rf2)
        avgrmask2 = self.avgpool2(maxmask)
        avgrf2 = avgrf2 / (avgrmask2 + 0.0000001)
        avgrf2 = avgrf2.squeeze()

        fetavg = torch.mean(pf1, 1)
        fetmax, _ = torch.max(pf1, 1)
        fetmin, _ = torch.min(pf1, 1)
        fetstd = torch.std(pf1, 1)
        at2 = torch.cat([fetavg, fetmax, fetmin, fetstd], 1)
        wei1 = self.cfe(at2)
        pf1 = pf1.squeeze()
        allf =   pf1 + wei1 * dstf + wei1 * avgrf2
        allf = self.ofe(allf)
        return allf

