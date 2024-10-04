import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DConv(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, kernel_size : int, stride : int, padding : int): 
        super().__init__() 
        
        self.kxk_conv = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.point_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        x = self.kxk_conv(x)
        output = self.point_conv(x)
        return output
    
class DConvBlock(nn.Module): 
    def __init__(self, input_channels : int, output_channels : int, kernel_size : int, stride : int, padding : int, activiation : str = 'relu'):
        super().__init__() 
        
        self.conv = DConv(input_channels, output_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        
        if activiation == 'relu': 
            self.activation = nn.ReLU()
        elif activiation == 'gelu': 
            self.activation = nn.GELU() 
        else: 
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv(x)
        x = self.batch_norm(x)
        output = self.activation(x)
        return output
    
class DownsampleConvBlock(nn.Module): 
    def __init__(self, channels : int, num_blocks : int = 3): 
        super().__init__()
        
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(*[DConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, activiation='relu') 
                                   for i in range(num_blocks)])
    
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        x = self.downsample(x)
        output = self.conv(x)
        return output, x
    
class UpsampleConvBlock(nn.Module): 
    def __init__(self, channels : int, num_blocks : int): 
        super().__init__() 

        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(*[DConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, activiation='relu') 
                                   for i in range(num_blocks)])
    
    def forward(self, x : torch.Tensor, residual : torch.Tensor): 
        x = x + residual
        x = self.upsample(x)
        output = self.conv(x)
        return output
    
class HourGlassModule(nn.Module):
    def __init__(self, channels : int, depth : int, num_blocks : int = 3):
        super().__init__()
        self.depth = depth
        
        self.downsample = nn.ModuleList([DownsampleConvBlock(channels, num_blocks) for _ in range(self.depth)])
        self.upsample = nn.ModuleList([UpsampleConvBlock(channels, num_blocks) for _ in range(self.depth)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residuals = []

        for i in range(self.depth):
            x, residual = self.downsample[i](x)
            residuals.append(residual)

        for i in range(self.depth):
            x = self.upsample[i](x, residuals[self.depth - i - 1])

        return x
    
class StackedHourGlass(nn.Module): 
    def __init__(self, input_channels : int, num_modules : int, num_depth : int, num_blocks : int, num_keypoints : int = 17): 
        super().__init__()
        
        self.feature_extractor = nn.Sequential(*[
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            DConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            DConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            DConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        ])
        
        self.hourGlassStem = nn.Sequential(*[
            HourGlassModule(256, depth=num_depth, num_blocks=num_blocks) for _ in range(num_modules)
        ])
        
        self.intermediate_convs = nn.Sequential(*[
            DConvBlock(256, 256, kernel_size=3, stride=1, padding=1, activiation='relu') 
            for _ in range(3)
        ])
        
        self.output_convs = nn.Sequential(*[nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2), 
                                            nn.Conv2d(256, num_keypoints, kernel_size=3, stride=1, padding=1)])
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:  
        x = self.feature_extractor(x)
        x = self.hourGlassStem(x)
        x = self.intermediate_convs(x)
        output = self.output_convs(x)
        
        batch_size, num_keypoints, height, width = output.shape
        output = output.view(batch_size, num_keypoints, -1)
        output = F.softmax(output, dim=2)
        output = output.view(batch_size, num_keypoints, height, width)
        return output

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = torch.nn.functional.silu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)

        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        self.se = SEBlock(expanded_channels, reduction=int(1 / se_ratio))

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))

        x = self.se(x)

        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            x = x + identity
        return x

class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(FusedMBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size // 2, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)

        self.se = SEBlock(expanded_channels, reduction=int(1 / se_ratio))

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = F.relu6(self.bn0(self.expand_conv(x)))

        x = self.se(x)

        x = self.bn1(self.project_conv(x))

        if self.use_residual:
            x = x + identity
        return x

class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        heatmaps = self.conv3(x)
        return heatmaps

class EfficientNetV2(nn.Module):
    def __init__(self, num_keypoints : int=17):
        super(EfficientNetV2, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )

        self.blocks = nn.Sequential(
            FusedMBConvBlock(24, 24, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.25),
            FusedMBConvBlock(24, 48, expand_ratio=4, kernel_size=3, stride=2, se_ratio=0.25),
            FusedMBConvBlock(48, 48, expand_ratio=4, kernel_size=3, stride=1, se_ratio=0.25),
            MBConvBlock(48, 64, expand_ratio=4, kernel_size=3, stride=2, se_ratio=0.25),
            MBConvBlock(64, 128, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25),
            MBConvBlock(128, 160, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25),
            MBConvBlock(160, 256, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25)
        )

        self.head = KeypointHead(256, num_keypoints)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
def get_EfficientNetV2(): 
    """
    Helper function for defining EfficientNetv2
    """
    return EfficientNetV2().to("cuda" if torch.cuda.is_available() else "cpu")