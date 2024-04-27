import torch
import math
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

##############################################Ours Loss########################################################################
def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
            #self.dc = MulticlassDiceLoss()
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        # net_output = torch.sigmoid(net_output)
        # print(np.unique(target.cpu().numpy()))
        # print('Loss input:',net_output.shape, target.shape, torch.unique(target))
        if len(torch.unique(target)) == 1:
            print('label wrong!!!')
            return torch.tensor(0.0,requires_grad=True)
        # if len(torch.unique(target)) != 2:
        #     print('label wrong!!!')
        #     return torch.tensor(0.0,requires_grad=True)

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            # print('DC Loss:',ce_loss, dc_loss)
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result

class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1, )

        # print(inp.size(), target.size())
        return super(CrossentropyND, self).forward(inp, target)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x))) # 把1，2维度去掉一个 [0,2,3,4]
        else:
            axes = list(range(2, len(shp_x))) # axes=[2,3,4]

        if self.apply_nonlin is not None:
            #x = self.apply_nonlin(x)
            soft = nn.Softmax(dim=1)
            x = soft(x)
        #print('softmax:',np.unique(x))
        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        #print(tp.size())
        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)
        #print(dc)
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            # print("loss:输入loss的"
            #       "data与label维度不对")
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            #print(gt.shape)
            y_onehot = gt
        else:
            gt = gt.long() # (2,1,128,128,128)
            # print('gt', gt.shape)
            y_onehot = torch.zeros(shp_x) # (2,3,128,128,128)
            # print(y_onehot.shape)
            if net_output.device.type == "cuda":
                #print(y_onehot.shape)
                y_onehot = y_onehot.cuda(net_output.device.index)
            # print(gt.shape)
            y_onehot.scatter_(1, gt, 1) # 包括背景，把不同维度的目标像素值标为1

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        #print(tp.shape, axes)
        tp = sum_tensor(tp, axes, keepdim=False)
        #print(tp.size())
        fp = sum_tensor(fp, axes, keepdim=False)
        #print(fp)
        fn = sum_tensor(fn, axes, keepdim=False)
        #print(fn)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        # print(len(x), len(y))
        # print(x[0].shape, y[0].shape, x[-1].shape, y[-1].shape)
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        # print(weights)
        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

# class DeepSupervisionWrapper(nn.Module):
#     def __init__(self, loss, weight_factors=None):
#         """
#         Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
#         inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
#         applied to each entry like this:
#         l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
#         If weights are None, all w will be 1.
#         """
#         super(DeepSupervisionWrapper, self).__init__()
#         self.weight_factors = weight_factors
#         self.loss = loss
#
#     def forward(self, *args):
#         for i in args:
#             assert isinstance(i, (tuple, list)), "all args must be either tuple or list, got %s" % type(i)
#             # we could check for equal lengths here as well but we really shouldn't overdo it with checks because
#             # this code is executed a lot of times!
#
#         if self.weight_factors is None:
#             weights = [1] * len(args[0])
#         else:
#             weights = self.weight_factors
#
#         # we initialize the loss like this instead of 0 to ensure it sits on the correct device, not sure if that's
#         # really necessary
#         # print(weights)
#         l = weights[0] * self.loss(*[j[0] for j in args])
#         for i, inputs in enumerate(zip(*args)):
#             if i == 0:
#                 continue
#             l += weights[i] * self.loss(*inputs)
#         return l
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):

            inp = inp.sum(int(ax))

    return inp

def loss_distill(u_prediction_1, u_prediction_2, T = 2):
    loss_a = 0.0
    u_prediction_1 = u_prediction_1/T
    u_prediction_2 = u_prediction_2/T
    u_prediction_1 = torch.sigmoid(u_prediction_1) / T
    u_prediction_2 = torch.sigmoid(u_prediction_2) / T

    # for i in range(u_prediction_2.size(1)):
    #     loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
    #                              Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))
    #     print('11',loss_a)

    loss_a = CE(u_prediction_1.clamp(1e-8, 1 - 1e-7),
                Variable(u_prediction_2.float(), requires_grad=False)) * T * T
    # loss_diff_avg = loss_a.mean().item()
    return loss_a

########################################################################################################################

CE = torch.nn.BCELoss()


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean().item()
    return loss_diff_avg


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean().item()
    return loss_diff_avg


def loss_diff3(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean().item()
    return loss_diff_avg


def loss_mask(u_prediction_1, u_prediction_2, critic_segs, T_m):
    # T_m = torch.mean(critic_segs)
    gen_mask = (critic_segs.squeeze(0) > T_m).float()
    loss_a = gen_mask * CE(u_prediction_1,
                                        Variable(u_prediction_2.float(), requires_grad=False))

    loss_diff_avg = loss_a.mean()

    return loss_diff_avg

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def disc_loss(pred, target, target_zeroes, target_ones):
    real_loss1 = CE(target, target_ones.float())
    fake_loss1 = CE(pred, target_zeroes.float())

    loss = (1/2) * (real_loss1 + fake_loss1)

    return loss


def gen_loss(pred, target_ones):
    fake_loss1 = CE(pred, target_ones.float())

    loss = fake_loss1

    return loss

################################desco##########################################################3
def dice_loss_weight(score,target,mask):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target*mask)
    y_sum = torch.sum(target * target*mask)
    z_sum = torch.sum(score * score*mask)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def wce(logits,target,weights,batch_size,H,W,D):
    # Calculate log probabilities
    logp = F.log_softmax(logits,dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W,D))
    # Multiply with weights
    weighted_logp = (logp * weights).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    #weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)
    weighted_loss = (weighted_logp.sum(1) - 0.00001) / (weights.view(batch_size, -1).sum(1) + 0.00001)
    # Average over mini-batch
    weighted_loss = -1.0*weighted_loss.mean()
    return weighted_loss

###################################PLN loss#######################################################

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def pixel_weighted_ce_loss(input, weights, target, bs):
    """
    input (B=2, C=2, 112, 112, 80)
    weights (B=2, 112, 112, 80)
    target (B=2, 112, 112, 80)
    """
    # Calculate log probabilities
    log_soft = F.log_softmax(input)
    # shape
    a, b, c = input.shape[-3], input.shape[-2], input.shape[-1]
    weights = weights.view(bs, 1, a, b, c)
    # Gather log probabilities with respect to target
    log_soft = log_soft.gather(1, target.view(bs, 1, a, b, c))
    # Multiply with weights
    weights = weights.float()
    weighted_log_soft = (log_soft * weights).view(bs, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_log_soft.sum(1) / weights.view(bs, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    # sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    # sum_filt = torch.ones([1, 1, *win]).to(device)
    sum_filt = torch.ones([1, 1, *win]).cuda()
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


##########################################################################################
def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes