"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
import spconv.pytorch as spconv


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


class ImageViewEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, num_layers=4, 
                index_interp="bilinear",
                feature_scale=0.5, use_first_pool=False):
        """Initialization
        
        Args:
            backbone: backbone Backbone network.
            num_layers: number of resnet layers to use, 1-5
            index_interp: Interpolation to use for indexing
            feature_scale: factor to scale all latent by. Useful (<1) if image
                is extremely large, to fit in memory.
            use_first_pool: if false, skips first maxpool layer to avoid downscaling image
                features too much (ResNet only)
        """
        super(ImageViewEncoder, self).__init__()

        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        self.norm_type = nn.BatchNorm2d

        # print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained, norm_layer=self.norm_type)
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.latent = None
        # self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        # )
        # self.latent (B, L, H, W)


    def index(self, uv, image_size=(200, 200)):
        """Get pixel-aligned image features at 2D image coordinates
        
        Args:
            :param uv (B, N, 2) image points (x,y)  # uv.shape torch.Size([4, 16384, 2])
            :param image_size image size, either (width, height) or single int.
                if not specified, assumes coords are in [-1, 1]
        
        Returns:
            :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):

            uv = uv.unsqueeze(2).type(torch.float32)  # (B, N, 1, 2)
            uv = 2.0 * uv / torch.tensor(image_size).to(uv.device) - 1.0

            # samples = F.grid_sample(
            #     self.latent.float(), # torch.Size([4, 512, 501, 500]) (B, C, H, W)
            #     uv,
            #     align_corners=True,
            #     mode="bilinear",
            #     padding_mode='zeros', 
            #     )# torch.Size([4, 512, 16384, 1])
            samples = grid_sample(self.latent.float(), uv)

            return samples[:, :, :, 0]  # [4, 512, 16384] (B, C, N)


    def forward(self, x):
        """For extracting ResNet's features.
        
            :param x image (B, C, H, W)
            :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        # x = x.to(device=self.latent.device)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)   # x.shape  torch.Size([4, 256, 63, 63])

        self.latents = latents
        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",
                align_corners=align_corners,
            ) # torch.Size([4, 64, 501, 500])

        # torch.Size([4, 512, 501, 500])
        self.latent = torch.cat(latents, dim=1) 

        return self.latent


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, num_layers=4, 
                index_interp="bilinear",
                feature_scale=0.5, use_first_pool=False):
        """Initialization
        
        Args:
            backbone: backbone Backbone network.
            num_layers: number of resnet layers to use, 1-5
            index_interp: Interpolation to use for indexing
            feature_scale: factor to scale all latent by. Useful (<1) if image
                is extremely large, to fit in memory.
            use_first_pool: if false, skips first maxpool layer to avoid downscaling image
                features too much (ResNet only)
        """
        super().__init__()

        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        self.norm_type = nn.BatchNorm2d

        # print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained, norm_layer=self.norm_type)
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.latent = None
        # self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        # )
        # self.latent (B, L, H, W)


    def index(self, uv, image_size=(512, 512)):
        """Get pixel-aligned image features at 2D image coordinates
        
        Args:
            :param uv (B, N, 2) image points (x,y)  # uv.shape torch.Size([4, 16384, 2])
            :param image_size image size, either (width, height) or single int.
                if not specified, assumes coords are in [-1, 1]
        
        Returns:
            :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):

            uv = uv.unsqueeze(2).type(torch.float32)  # (B, N, 1, 2)
            # uv = 2.0 * uv / image_size.cuda().float() - 1.0
            uv = 2.0 * uv / image_size.clone().detach().to(uv.device) - 1.0 #torch.tensor(image_size).to(uv.device) - 1.0
            # uv = torch.flip(uv, (-1,))
            # uv[:,:,:,0] = 2 * uv[:,:,:,0] / image_size[0] - 1.
            # uv[:,:,:,1] = 2 * uv[:,:,:,1] / image_size[1] - 1.
            samples = grid_sample(self.latent.float(), uv)
            # samples = F.grid_sample(
            #     self.latent.float(), # torch.Size([4, 512, 501, 500]) (B, C, H, W)
            #     uv,
            #     align_corners=True,
            #     mode="bilinear",
            #     padding_mode='zeros', 
            #     )# torch.Size([4, 512, 16384, 1])

            return samples[:, :, :, 0]  # [4, 512, 16384] (B, C, N)


    def forward(self, x):
        """For extracting ResNet's features.
        
            :param x image (B, C, H, W)
            :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        # x = x.to(device=self.latent.device)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)   # x.shape  torch.Size([4, 256, 63, 63])

        self.latents = latents
        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",
                align_corners=align_corners,
            ) # torch.Size([4, 64, 501, 500])

        # torch.Size([4, 512, 501, 500])
        self.latent = torch.cat(latents, dim=1) 

        return self.latent


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=512):
        """Initialization

        Args: 
            :param backbone Backbone network. Assumes it is resnet*
                e.g. resnet34 | resnet50
            :param num_layers number of resnet layers to use, 1-5
            :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        # self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        self.latent = None
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(latent_size, 512)

    def index(self, uv, image_size=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        # x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent


class SparseConvNet(nn.Module):
    """Find the corresponding 3D feature of query point along the ray
    
    Attributes:
        conv: sparse convolutional layer 
        down: sparse convolutional layer with downsample 
    """
    def __init__(self, num_layers=2):
        super(SparseConvNet, self).__init__()
        self.num_layers = num_layers

        self.conv0 = double_conv(3, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

        self.channel = 32

    def forward(self, x, point_normalied_coords):
        """Find the corresponding 3D feature of query point along the ray.

        Args:
            x: Sparse Conv Tensor
            point_normalied_coords: Voxel grid coordinate, integer normalied to [-1, 1]
        
        Returns:
            features: Corresponding 3D feature of query point along the ray
        """
        features = []

        net = self.conv0(x)
        net = self.down0(net)

        point_normalied_coords = point_normalied_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if self.num_layers > 1:
            net = self.conv1(net)
            net1 = net.dense()
            # torch.Size([1, 32, 1, 1, 4096])
            feature_1 = F.grid_sample(net1, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_1)
            self.channel = 32
            net = self.down1(net)
        
        if self.num_layers > 2:
            net = self.conv2(net)
            net2 = net.dense()
            # torch.Size([1, 64, 1, 1, 4096])
            feature_2 = F.grid_sample(net2, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_2)
            self.channel = 64
            net = self.down2(net)
        
        if self.num_layers > 3:
            net = self.conv3(net)
            net3 = net.dense()
            # 128
            feature_3 = F.grid_sample(net3, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_3)
            self.channel = 128
            net = self.down3(net)
        
        if self.num_layers > 4:
            net = self.conv4(net)
            net4 = net.dense()
            # 256
            feature_4 = F.grid_sample(net4, point_normalied_coords, padding_mode='zeros', align_corners=True)
            features.append(feature_4)

        features = torch.cat(features, dim=1)
        # TODO, why so much dimention ???
        # B*C*D*H*W or B*D*H*W*C
        # B*N*C or B*C*N(num of query point)
        # torch.Size([1, 352, 4096])
        features = features.view(features.size(0), self.channel, features.size(4)).squeeze(0).transpose(0,1)

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    tmp = spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key)
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())

