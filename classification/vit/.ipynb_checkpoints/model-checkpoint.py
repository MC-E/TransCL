"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS
import numpy as np
import os

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        weights_path = None
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)   # patch sizes
        gh, gw = h // fh, w // fw    # number of patches
        seq_len = gh * gw
#         print('seq_len:', seq_len,h,w,fh,fw)
#         exit(0)

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)
        
        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = 1000 #PRETRAINED_MODELS[name]['num_classes'] #edit for new
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
#             print(image_size,pretrained_image_size)
#             exit(0)
            load_pretrained_weights(
                self, name, 
                weights_path= weights_path,#'pretrain/B_32_imagenet1k.pth',
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )
            print('load_first_conv:',in_channels == pretrained_num_channels,'load_fc:',num_classes == pretrained_num_classes)
#         self.idx=[0]*1000
#         self.dict_={}
        
        # # Modify model as specified. NOTE: We do not do this earlier because 
        # # it's easier to load only part of a pretrained model in this manner.
        # if in_channels != 3:
        #     self.embedding = nn.Conv2d(in_channels, patches, kernel_size=patches, stride=patches)
        # if num_classes is not None and num_classes != num_classes_init:
        #     self.fc = nn.Linear(dim, num_classes)
    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x
#     def forward(self, x):#,y):
#         """Breaks image into patches, applies transformer, applies MLP head.

#         Args:
#             x (tensor): `b,c,fh,fw`
#         """
# #         blocks=F.unfold(x,kernel_size=32,stride=32).permute(0, 2, 1).contiguous()
# #         np.save('feature_TCL_org.npy',blocks[0,:,:].cpu().data.numpy())
# #         exit(0)
#         b, c, fh, fw = x.shape
#         x = self.patch_embedding(x)  # b,d,gh,gw
#         x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
#         if hasattr(self, 'class_token'):
#             x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
#         if hasattr(self, 'positional_embedding'): 
#             x = self.positional_embedding(x)  # b,gh*gw+1,d 
#         x = self.transformer(x)  # b,gh*gw+1,d
# #         np.save('feature_vit_b16_2.npy',x[2,:,:].cpu().data.numpy())

# #         for i in range(len(y)):
# #             class_=str(y[i])
# #             if os.path.isdir(os.path.join('feature_rp/TCL_2',class_))==False:
# #                 os.mkdir(os.path.join('feature_rp/TCL_2',class_))
# #             np.save(os.path.join('feature_rp/TCL_2',class_,'feature_TCL_%d.npy'%self.idx[y[i]]),x[i,:,:].cpu().data.numpy())
# #             np.save(os.path.join('feature_rp/TCL_2',class_,'feature_TCL_%d_org.npy'%self.idx[y[i]]),blocks[i,:,:].cpu().data.numpy())
# #             self.idx[y[i]]=self.idx[y[i]]+1
            
# #         for i in range(len(y)):
# #             class_=str(y[i])
# #             if os.path.isdir(os.path.join('feature_rp/TCL',class_))==False:
# #                 os.mkdir(os.path.join('feature_rp/TCL',class_))
# #             np.save(os.path.join('feature_rp/TCL',class_,'feature_TCL_%d.npy'%self.idx[y[i]]),x[i,:,:].cpu().data.numpy())
# #             self.idx[y[i]]=self.idx[y[i]]+1
        
# #         print(x[:,0].shape)
# #         exit(0)
#         if hasattr(self, 'pre_logits'):
#             x = self.pre_logits(x)
#             x = torch.tanh(x)
#         if hasattr(self, 'fc'):
#             x = self.norm(x)[:, 0]  # b,d
#             x = self.fc(x)  # b,num_classes
#         return x

