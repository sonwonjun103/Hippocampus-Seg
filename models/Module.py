import torch
import torch.nn as nn

from scipy.ndimage import distance_transform_edt

def get_binary_map(volume, threshold):
    copy_volume = volume.clone()

    copy_volume[copy_volume >= threshold] = 1
    copy_volume[copy_volume <= threshold] = 0

    return copy_volume

class Distance_map(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, threshold):
        binary_x = get_binary_map(x, threshold)
        background = 1-binary_x
        foreground = binary_x

        distance_background = distance_transform_edt(background)
        distance_foreground = distance_transform_edt(foreground)

        distance_map = 1- (distance_background + distance_foreground)

        return distance_map

class convblock(nn.Module):
    def __init__(self,
                 in_channel,
                 kernel_size,
                 padding=0,
                 dilate=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channel, 
                      in_channel, 
                      kernel_size=kernel_size,
                      dilation=dilate,
                      padding=padding),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x) 

class dilatedConv(nn.Module):
    def __init__(self,
                 in_channel):
        super().__init__()
        self.conv1x1x1 = convblock(in_channel=in_channel,
                                   kernel_size=3,
                                   padding=1,
                                   dilate=1)
        self.conv3x3x3 = convblock(in_channel=in_channel,
                                   kernel_size=3,
                                   padding=3,
                                   dilate=3)
        self.conv5x5x5 = convblock(in_channel=in_channel,
                                   kernel_size=3,
                                   padding=5,
                                   dilate=5)
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel * 3, in_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        conv1 = self.conv1x1x1(x)
        conv3 = self.conv3x3x3(x)
        conv5 = self.conv5x5x5(x)

        out = self.conv(torch.concat([conv1, conv3, conv5], dim=1))

        return out

class Module(nn.Module):
    def __init__(self,
                 in_channel):
        super().__init__()

        self.in_channel = in_channel
        self.dilate = dilatedConv(self.in_channel)
        self.distance_map = Distance_map()

    def forward(self, seg, edge): 
        dedge = self.dilate(edge)

        # 뭔가 distance map을 구하고 싶은데
        seg_distance_map = torch.from_numpy(self.distance_map(seg, threshold=0.5))

        edge_output = seg_distance_map * dedge
        seg_output = dedge + seg

        return seg_output, edge_output

if __name__=='__main__':
    sample1 = torch.randn(4, 64, 24, 32, 32)
    sample2 = torch.randn(4, 64, 24, 32, 32)

    model = Module(64)
    pred = model(sample1, sample2)