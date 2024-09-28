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
        x - self.batch_norm(x)
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
        x+=residual
        x = self.upsample(x)
        output = self.conv(x)
        return output
    
class HourGlassModule(nn.Module):
    def __init__(self, channels : int, depth : int, num_blocks : int = 3):
        super().__init__()
        self.depth = depth
        self.downsample = []
        self.upsample = []
        
        for _ in range(self.depth):
            self.downsample.append(DownsampleConvBlock(channels, num_blocks))
            self.upsample.append(UpsampleConvBlock(channels, num_blocks))
        
        self.downsample = nn.ModuleList(self.downsample)
        self.upsample = nn.ModuleList(self.upsample)

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