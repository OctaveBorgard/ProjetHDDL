import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Optional, Tuple

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(
            f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}"
        )


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        non_linearity: str = "swish",
        skip_connection: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,)           
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = skip_connection
        self.activation_fn = get_activation(non_linearity)
    
        if skip_connection:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(
        self,
        input_tensor: torch.Tensor,
        res_hidden_states: Tuple[torch.Tensor, ...] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if res_hidden_states is not None:
            input_tensor = torch.cat([input_tensor, res_hidden_states], dim=1)

        hidden_states = self.conv1(input_tensor)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.conv2(hidden_states)
        if self.skip_connection:
            residual = self.conv_skip(input_tensor)
            hidden_states += residual
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        return hidden_states


class Unet_Segmenter(torch.nn.Module):
    in_channels = 3
    input_shape = (224, 244)
    num_classes = 3
    block_out_channels = (16, 64, 128)
    
    def __init__(self,
                layers_per_block: int=2,
                non_linearity: str="swish",
                skip_connection: bool=True,
                center_input_sample: bool=True,
                ):
        super().__init__()

        self.center_input_sample = center_input_sample
        self.conv_in = nn.Conv2d(self.in_channels, self.block_out_channels[0], kernel_size=3, padding="same")
        self.norm_in = nn.BatchNorm2d(self.block_out_channels[0])
        self.activation_fn = get_activation(non_linearity)


        self.down_blocks = nn.ModuleList()
        self.mid_blocks = None
        self.up_blocks = nn.ModuleList()

        # Downsampling blocks
        prev_out_channels = self.block_out_channels[0]
        for out_channels in self.block_out_channels[:-1]:
            for _ in range(layers_per_block):
                resnet_block = ResnetBlock2D(
                    in_channels=prev_out_channels,
                    out_channels=out_channels,
                    non_linearity=non_linearity,
                    skip_connection=skip_connection,
                )
                self.down_blocks.append(resnet_block)
                prev_out_channels = out_channels
            self.down_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Middle block
        self.mid_blocks = ResnetBlock2D(
            in_channels=prev_out_channels,
            out_channels=self.block_out_channels[-1],
            non_linearity=non_linearity,
            skip_connection=skip_connection,
        )
        prev_out_channels = self.block_out_channels[-1]
        # Upsampling blocks
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        for out_channels in reversed_block_out_channels[1:]:
            self.up_blocks.append(nn.ConvTranspose2d(prev_out_channels, out_channels, kernel_size=2, stride=2))
            prev_out_channels = out_channels
            for _ in range(layers_per_block):
                resnet_block = ResnetBlock2D(
                    in_channels=prev_out_channels*2, # due to skip connection
                    out_channels=out_channels,
                    non_linearity=non_linearity,
                    skip_connection=skip_connection,
                )
                self.up_blocks.append(resnet_block)
                prev_out_channels = out_channels
            
        self.conv_out = nn.Conv2d(prev_out_channels, self.num_classes, kernel_size=3, padding="same")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.center_input_sample:
            x = 2.0 * x - 1.0  # scale to [-1, 1]

        hidden_states = self.conv_in(x)
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        skip_connections = []
        # Downsampling
        for i, block in enumerate(self.down_blocks):
            if isinstance(block, ResnetBlock2D):
                hidden_states = block(hidden_states)
                skip_connections.append(hidden_states)
            else:  # pooling
                hidden_states = block(hidden_states)
        
        # Middle block
        hidden_states = self.mid_blocks(hidden_states)

        # Upsampling
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, nn.ConvTranspose2d):
                hidden_states = block(hidden_states)
            else:  # ResnetBlock2D
                skip_connection = skip_connections.pop()
                hidden_states = block(hidden_states, skip_connection)
        
        logits = self.conv_out(hidden_states)
        return logits


def EfficientNetB0(num_classes: int, weights=EfficientNet_B0_Weights.DEFAULT):
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


class Unet_Segmenterv2(torch.nn.Module):
    """
    Adaptive U-Net Segmenter that can handle any input shape.
    
    The architecture automatically adapts to input dimensions at inference time,
    ensuring skip connections work correctly regardless of input resolution.
    
    Args:
        in_channels (int, optional): Number of input channels. Default: 3
        num_classes (int, optional): Number of output classes. Default: 3
        layers_per_block (int, optional): Number of ResNet blocks per layer. Default: 2
        non_linearity (str, optional): Activation function name. Default: "swish"
        skip_connection (bool, optional): Whether to use skip connections. Default: True
        center_input_sample (bool, optional): Whether to scale input to [-1, 1]. Default: True
        block_out_channels (tuple, optional): Output channels for each block level. Default: (16, 64, 128)
    """
    
    def __init__(self,
                in_channels: int = 3,
                num_classes: int = 3,
                layers_per_block: int = 2,
                non_linearity: str = "swish",
                skip_connection: bool = True,
                center_input_sample: bool = True,
                block_out_channels: Tuple[int, ...] = (16, 64, 128),
                ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.center_input_sample = center_input_sample
        self.block_out_channels = block_out_channels
        self.skip_connection = skip_connection
        self.non_linearity = non_linearity
        self.layers_per_block = layers_per_block
        
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding="same")
        self.norm_in = nn.BatchNorm2d(block_out_channels[0])
        self.activation_fn = get_activation(non_linearity)

        self.down_blocks = nn.ModuleList()
        self.mid_blocks = None
        self.up_blocks = nn.ModuleList()

        # Downsampling blocks
        prev_out_channels = block_out_channels[0]
        for out_channels in block_out_channels[:-1]:
            for _ in range(layers_per_block):
                resnet_block = ResnetBlock2D(
                    in_channels=prev_out_channels,
                    out_channels=out_channels,
                    non_linearity=non_linearity,
                    skip_connection=skip_connection,
                )
                self.down_blocks.append(resnet_block)
                prev_out_channels = out_channels
            self.down_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Middle block
        self.mid_blocks = ResnetBlock2D(
            in_channels=prev_out_channels,
            out_channels=block_out_channels[-1],
            non_linearity=non_linearity,
            skip_connection=skip_connection,
        )
        prev_out_channels = block_out_channels[-1]
        
        # Upsampling blocks
        reversed_block_out_channels = list(reversed(block_out_channels))
        for out_channels in reversed_block_out_channels[1:]:
            self.up_blocks.append(nn.ConvTranspose2d(prev_out_channels, out_channels, kernel_size=2, stride=2))
            prev_out_channels = out_channels
            for _ in range(layers_per_block):
                resnet_block = ResnetBlock2D(
                    in_channels=prev_out_channels*2,  # due to skip connection concatenation
                    out_channels=out_channels,
                    non_linearity=non_linearity,
                    skip_connection=skip_connection,
                )
                self.up_blocks.append(resnet_block)
                prev_out_channels = out_channels
            
        self.conv_out = nn.Conv2d(prev_out_channels, num_classes, kernel_size=3, padding="same")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that adapts to input shape.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where H and W can be any size.
            
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes, H, W) matching input spatial dimensions.
        """
        if self.center_input_sample:
            x = 2.0 * x - 1.0  # scale to [-1, 1]

        # Initial convolution
        hidden_states = self.conv_in(x)
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        skip_connections = []
        
        # Downsampling path
        for block in self.down_blocks:
            if isinstance(block, ResnetBlock2D):
                hidden_states = block(hidden_states)
                skip_connections.append(hidden_states)
            else:  # MaxPool2d
                hidden_states = block(hidden_states)
        
        # Middle block
        hidden_states = self.mid_blocks(hidden_states)

        # Upsampling path
        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                hidden_states = block(hidden_states)
            else:  # ResnetBlock2D
                skip_connection = skip_connections.pop()
                
                # Adaptive skip connection: handle size mismatches due to odd dimensions
                if hidden_states.shape[-2:] != skip_connection.shape[-2:]:
                    # Crop or pad skip_connection to match hidden_states size
                    h_diff = skip_connection.shape[-2] - hidden_states.shape[-2]
                    w_diff = skip_connection.shape[-1] - hidden_states.shape[-1]
                    
                    if h_diff > 0 or w_diff > 0:
                        # Crop skip connection
                        h_crop = h_diff // 2
                        w_crop = w_diff // 2
                        skip_connection = skip_connection[
                            :, :,
                            h_crop:skip_connection.shape[-2]-h_crop if h_crop > 0 else skip_connection.shape[-2],
                            w_crop:skip_connection.shape[-1]-w_crop if w_crop > 0 else skip_connection.shape[-1]
                        ]
                
                hidden_states = block(hidden_states, skip_connection)
        
        logits = self.conv_out(hidden_states)
        return logits