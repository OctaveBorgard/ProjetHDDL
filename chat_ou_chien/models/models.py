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
    block_out_channels = (64, 128, 256, 512, 1024)
    
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