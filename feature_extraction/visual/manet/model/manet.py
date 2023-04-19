from manet.model.attention import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MulScaleBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MulScaleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv1_2_3 = conv3x3(scale_width, scale_width)
        self.bn1_2_3 = norm_layer(scale_width)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)

        self.conv2_2_1 = conv3x3(scale_width, scale_width)
        self.bn2_2_1 = norm_layer(scale_width)
        self.conv2_2_2 = conv3x3(scale_width, scale_width)
        self.bn2_2_2 = norm_layer(scale_width)
        self.conv2_2_3 = conv3x3(scale_width, scale_width)
        self.bn2_2_3 = norm_layer(scale_width)
        self.conv2_2_4 = conv3x3(scale_width, scale_width)
        self.bn2_2_4 = norm_layer(scale_width)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        ##########################################################
        out_1_1 = self.conv1_2_1(sp_x[0])
        out_1_1 = self.bn1_2_1(out_1_1)
        out_1_1_relu = self.relu(out_1_1)
        out_1_2 = self.conv1_2_2(out_1_1_relu + sp_x[1])
        out_1_2 = self.bn1_2_2(out_1_2)
        out_1_2_relu = self.relu(out_1_2)
        out_1_3 = self.conv1_2_3(out_1_2_relu + sp_x[2])
        out_1_3 = self.bn1_2_3(out_1_3)
        out_1_3_relu = self.relu(out_1_3)
        out_1_4 = self.conv1_2_4(out_1_3_relu + sp_x[3])
        out_1_4 = self.bn1_2_4(out_1_4)
        output_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)

        out_2_1 = self.conv2_2_1(sp_x[0])
        out_2_1 = self.bn2_2_1(out_2_1)
        out_2_1_relu = self.relu(out_2_1)
        out_2_2 = self.conv2_2_2(out_2_1_relu + sp_x[1])
        out_2_2 = self.bn2_2_2(out_2_2)
        out_2_2_relu = self.relu(out_2_2)
        out_2_3 = self.conv2_2_3(out_2_2_relu + sp_x[2])
        out_2_3 = self.bn2_2_3(out_2_3)
        out_2_3_relu = self.relu(out_2_3)
        out_2_4 = self.conv2_2_4(out_2_3_relu + sp_x[3])
        out_2_4 = self.bn2_2_4(out_2_4)
        output_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=1)

        out = output_1 + output_2

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttentionBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MANet(nn.Module):

    def __init__(self, block_b, block_m, block_a, layers, num_classes=12666):
        super(MANet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0])
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)

        # In this branch, each BasicBlock replaced by AttentiveBlock.
        self.layer3_1_p1 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p1 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p2 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p2 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p3 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p3 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p4 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p4 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        # In this branch, each BasicBlock replaced by MulScaleBlock.
        self.layer3_2 = self._make_layer(block_m, 128, 256, layers[2], stride=2)
        self.layer4_2 = self._make_layer(block_m, 256, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_1 = nn.Linear(512, num_classes)
        self.fc_2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x, return_embedding=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # branch 1 ############################################
        patch_11 = x[:, :, 0:14, 0:14]
        patch_12 = x[:, :, 0:14, 14:28]
        patch_21 = x[:, :, 14:28, 0:14]
        patch_22 = x[:, :, 14:28, 14:28]

        branch_1_p1_out = self.layer3_1_p1(patch_11)
        branch_1_p1_out = self.layer4_1_p1(branch_1_p1_out)

        branch_1_p2_out = self.layer3_1_p2(patch_12)
        branch_1_p2_out = self.layer4_1_p2(branch_1_p2_out)

        branch_1_p3_out = self.layer3_1_p3(patch_21)
        branch_1_p3_out = self.layer4_1_p3(branch_1_p3_out)

        branch_1_p4_out = self.layer3_1_p4(patch_22)
        branch_1_p4_out = self.layer4_1_p4(branch_1_p4_out)

        branch_1_out_1 = torch.cat([branch_1_p1_out, branch_1_p2_out], dim=3)
        branch_1_out_2 = torch.cat([branch_1_p3_out, branch_1_p4_out], dim=3)
        branch_1_out = torch.cat([branch_1_out_1, branch_1_out_2], dim=2)

        branch_1_out = self.avgpool(branch_1_out)
        branch_1_out_embedding = torch.flatten(branch_1_out, 1)
        branch_1_out = self.fc_1(branch_1_out_embedding)

        # branch 2 ############################################
        branch_2_out = self.layer3_2(x)
        branch_2_out = self.layer4_2(branch_2_out)
        branch_2_out = self.avgpool(branch_2_out)
        branch_2_out_embedding = torch.flatten(branch_2_out, 1)
        branch_2_out = self.fc_2(branch_2_out_embedding)

        if return_embedding:
            return torch.cat([branch_1_out_embedding, branch_2_out_embedding], dim=1)
        else:
            return branch_1_out, branch_2_out

    def forward(self, x, return_embedding=False):
        return self._forward_impl(x, return_embedding)


def manet(**kwargs):
    return MANet(block_b=BasicBlock, block_m=MulScaleBlock, block_a=AttentionBlock, layers=[2, 2, 2, 2], **kwargs)

if __name__ == "__main__":
    model = manet()
    # print(sum(param.numel() for param in model.parameters()) / 1e6)
    # print(model)
    import torch
    x = torch.rand(1,3,224,224)
    y = model(x)
