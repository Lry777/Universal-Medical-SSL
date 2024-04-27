from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3
from networks.VNet import VNet, MCNet3d_v1, MCNet3d_v2, CAML3d_v1
from networks.TAC import TACNet
from networks.newModel import newNet3d

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train", patch_size = None, **kwargs):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "tac" and mode == "train":
        net = TACNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v1" and mode == "train":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "caml3d_v1" and mode == "train":
        net = CAML3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
    elif net_type == "newNet3d" and mode == "train":
        net = newNet3d(in_channels=in_chns, out_channels=class_num, normalization='instancenorm', has_dropout=True, patch_size = patch_size).cuda()



    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v1" and mode == "test":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "newNet3d" and mode == "test":
        net = newNet3d(in_channels=in_chns, out_channels=class_num, normalization='instancenorm', has_dropout=False, patch_size = patch_size).cuda()
    elif net_type == "tac" and mode == "test":
        net = TACNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "caml3d_v1" and mode == "test":
        net = CAML3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, **kwargs).cuda()

    return net