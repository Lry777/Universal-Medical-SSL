import torch
from medclip import MedCLIPModel, MedCLIPTextModel
from medclip import MedCLIPProcessor

# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# _tokenizer = _Tokenizer
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from .utils import make_fig
# from .STUNet import STUNet
# from .utils import load_pretrained_weights


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    
    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=32, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv1 = nn.Conv3d(n_filters*2, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters*4, n_classes, 1, padding=0)
        self.out_conv3 = nn.Conv3d(n_filters*8, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4


        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return [out_seg, self.out_conv1(x8), self.out_conv2(x7), self.out_conv3(x6)]
        # return x9


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class newNet3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=2,
                 n_filters=16, normalization='none',
                 has_dropout=False, has_residual=False,
                 patch_size = (112, 112, 80)):
        super(newNet3d, self).__init__()
        self.image_encode = Encoder(in_channels, out_channels, n_filters,normalization,  has_dropout, has_residual)
        self.image_aug_encode = Encoder(in_channels, out_channels, n_filters, normalization, has_dropout, has_residual)
        # self.image_aug_encode = STUNet(input_channels = in_channels, num_classes= out_channels)
        # self.image_aug_encode.inference_apply_nonlin = lambda x: F.softmax(x, 1)
        # load_pretrained_weights(self.image_aug_encode, fname='/home/lry/Code/Med_Pretrain/HybridMIM/Pretrain/weights/base_ep4k.model')

        self.text_encoder = BioTextEncoder(bert_type='emilyalsentzer/Bio_ClinicalBERT', N=(patch_size[0]*patch_size[1]*patch_size[2])//(16*16*16))
        self.Joint_Gate = Joint_Gate(n_class=out_channels, i_channels=256, o_channels=256)
        # self.half_cha = nn.Sequential(
        #     nn.Conv3d(256, 256*2, 1),
        #     # nn.InstanceNorm3d(256),
        #     # nn.LeakyReLU()
        # )


        self.decoder1 = Decoder(in_channels, out_channels, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(in_channels, out_channels, n_filters, normalization, has_dropout, has_residual, 1)


    def forward(self, input, input_aug, text=None):

        features = self.image_encode(input)

        # with torch.no_grad():
        features_aug = self.image_aug_encode(input_aug)

        # print( len(features),features[-1].shape, features[-2].shape, features[-3].shape)
        # print(len(features_aug), features_aug[-1].shape, features_aug[-2].shape, features_aug[-3].shape)

        rate = [0,0,0,1, 1]
        features = [features[i]+item *rate[i] for i, item in enumerate(features_aug)]

        # features[-1] = features[-1] + self.half_cha(features_aug[-1])

        if text is not None:
            # b, c, h, w, d = features[-1].shape
            text_embedding = self.text_encoder(text)  # (C, N)
            # make_fig(text_embedding)
            image_tocken = features[-1]
            # image_tocken = self.mix(torch.cat([features[-1], features_aug[-1]], dim=1))
            features[-1] = self.Joint_Gate(text_embedding, image_tocken)
            ####### up text ################

            out_seg1 = self.decoder1(features)
            out_seg2 = self.decoder2(features)

            # print(out_seg1.shape,out_seg2.shape)
            # make_fig(out_seg1, out_seg2)
        else:
            # print('no text')
            out_seg1 = self.decoder1(features)
            out_seg2 = self.decoder2(features)

        # return seg1, seg2
        # print(out_seg1, out_seg2)
        return out_seg1, out_seg2

class Joint_Gate(nn.Module):
    def __init__(self, n_class = 2, i_channels=256, o_channels=256):
        super(Joint_Gate, self).__init__()

        self.text_t = nn.Sequential(
            nn.Linear(n_class, i_channels),
            nn.LayerNorm(i_channels),
            nn.LeakyReLU()
        )

        self._w = nn.Sequential(
            nn.Conv3d(i_channels, o_channels, 1),
            nn.InstanceNorm3d(o_channels),
        )
        self.relu = nn.LeakyReLU()
        self.psi = nn.Sequential(
            nn.Conv3d(o_channels, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.mix_MLP = nn.Sequential(
            nn.Conv3d(i_channels*2, o_channels, 1),
            nn.InstanceNorm3d(o_channels),
            nn.LeakyReLU()
        )

    def forward(self, text_embedding, img_tocken):
        b, c, h, w, d = img_tocken.shape
        text_embedding = text_embedding.unsqueeze(0).expand(b, -1, -1).transpose(1, 2)  # [B, N, C]
        text_tocken = self.text_t(text_embedding)
        text_tocken = text_tocken.transpose(1, 2).view(b, c, h, w, d)

        w1 = self._w(img_tocken)
        w2 = self._w(text_tocken)
        psi = self.relu(w1+w2)
        psi = self.psi(psi)

        con_emb = torch.concat([img_tocken, text_tocken], dim=1)
        con_feature = self.mix_MLP(con_emb)
        # print(con_emb.shape)

        # return con_feature*psi

        return con_feature * psi


class BioTextEncoder(nn.Module):
    def __init__(self, bert_type, N, B=2):
        super().__init__()
        bert_type = './pretrained/Bio_ClinicalBERT'
        self.text_model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.projection_head = nn.Linear(768, N, bias=False)
        self.positional_embedding = nn.Parameter(torch.empty(B, 768))

        # self.text_model = MedCLIPModel()
        # self.text_model.from_pretrained()

        self.initialize_paramters()

    def organ2tokens(self, organ_names):
        text_list = ['A computerized tomography of a {}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")

        # promptlearner = PromptLearner(classnames=organ_names, model=self.text_model, tokenizer=self.tokenizer)
        # promot = promptlearner()
        # print(tokens['input_ids'].shape, tokens['attention_mask'].shape, tokens["token_type_ids"].shape)

        for key in tokens.keys():
            tokens[key] = tokens[key].cuda()
        return tokens

    def initialize_paramters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, text):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        with torch.no_grad():
            tokens = self.organ2tokens(text)
            # print(len(tokens), tokens)
            # text_outputs = self.text_model(**tokens)
            # text_outputs = self.text_model.encode_text(input_ids=tokens['input_ids'], attention_mask=tokens['input_ids'])
            # print(text_outputs.shape)
            text_outputs = self.text_model(**tokens)
            text_e = text_outputs.pooler_output

        # print(text_e.shape, self.positional_embedding.shape)
        text_e = text_e + self.positional_embedding
        # print(text_e.shape, text_e)
        # make_fig(text_e)
        text_embedding = self.projection_head(text_e)
        # uptext_embedding = self.projection_head2(text_e)
        # print(text_embedding.shape)
        return text_embedding


# class PromptLearner(nn.Module):
#     def __init__(self, classnames, text_model, tokenize, CSC = True):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = 0
#         ctx_init = 'A computerized tomography of a '
#         dtype = text_model.dtype  # clip_model 的数据类型
#         ctx_dim = 768
#         # clip_imsize = clip_model.visual.input_resolution
#         # cfg_imsize = cfg.INPUT.SIZE[0]  # 输入图像的大小
#         # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
#
#         if ctx_init:
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = tokenize(ctx_init)
#
#             with torch.no_grad():
#                 # embedding = clip_model.token_embedding(prompt).type(dtype) # [bs, n_ctx, d_model]
#                 embedding = text_model(**prompt).pooler_output # (B, N)
#
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#
#         else:
#             if CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#         self.ctx = nn.Parameter(ctx_vectors)
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#
#         with torch.no_grad():
#             # embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#             embedding = text_model(**tokenized_prompts).pooler_output  # (B, N)
#
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim) 前缀 "a photo of a" 的嵌入向量
#                     ctx,  # (n_cls, n_ctx, dim) 上下文向量
#                     suffix,  # (n_cls, *, dim) 后缀 "car" 的嵌入向量
#                 ],class PromptLearner(nn.Module):
#     def __init__(self, classnames, text_model, tokenize, CSC = True):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = 0
#         ctx_init = 'A computerized tomography of a '
#         dtype = text_model.dtype  # clip_model 的数据类型
#         ctx_dim = 768
#         # clip_imsize = clip_model.visual.input_resolution
#         # cfg_imsize = cfg.INPUT.SIZE[0]  # 输入图像的大小
#         # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
#
#         if ctx_init:
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = tokenize(ctx_init)
#
#             with torch.no_grad():
#                 # embedding = clip_model.token_embedding(prompt).type(dtype) # [bs, n_ctx, d_model]
#                 embedding = text_model(**prompt).pooler_output # (B, N)
#
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#
#         else:
#             if CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#         self.ctx = nn.Parameter(ctx_vectors)
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#
#         with torch.no_grad():
#             # embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#             embedding = text_model(**tokenized_prompts).pooler_output  # (B, N)
#
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim) 前缀 "a photo of a" 的嵌入向量
#                     ctx,  # (n_cls, n_ctx, dim) 上下文向量
#                     suffix,  # (n_cls, *, dim) 后缀 "car" 的嵌入向量
#                 ],
#                 dim=1,
#             )
#
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,  # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#
#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i = ctx[i: i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,  # (1, name_len, dim)
#                         ctx_i,  # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#
#         else:
#             raise ValueError
#
#         return prompts


if __name__ == '__main__':
    text = ['Background','Left Atrial']
    # promptlearnercc = PromptLearner(text, clip)
