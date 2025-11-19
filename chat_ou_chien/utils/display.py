"""
Some useful utilities for displaying
"""

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from typing import List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore", module=r"matplotlib\..*")

_TensorArray = Union[Tensor, np.ndarray]

def to_numpy_image(input):
    if isinstance(input, Tensor):
        if input.dim() == 3:
            return input.detach().cpu().permute(1, 2, 0).numpy()
        elif input.dim() == 4:
            return input.detach().cpu().permute(0, 2, 3, 1).numpy()

    elif isinstance(input, np.ndarray):
        print("Warning: input is already a numpy array")
        return input
    else:
        raise ValueError(f"Cannot convert {type(input)} to numpy image")


def show_images(
    imgs: Union[_TensorArray, List[_TensorArray], Tuple[_TensorArray]],
    title: List[str] = None,
    suptitle: str = None,
    ncols: int = None,
    colorbar: Optional[bool] = False,
    colorbar_geq: Optional[bool] = False,  # NEW OPTION
    cmap: str = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    savename: Optional[str] = None,
    figsize: Optional[int] = 3,
    interpolation: Optional[str] = None,
):
    if not isinstance(imgs, np.ndarray):
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            imgs = [to_numpy_image(img) for img in imgs]
            B = len(imgs)
            C = imgs[0].shape[-1]
        elif isinstance(imgs, torch.Tensor):
            imgs = to_numpy_image(imgs)
            B = imgs.shape[0] if imgs.ndim==4 else 1
            C = imgs.shape[-1]
        else:
            raise TypeError("imgs must be a list, tuple, torch.tensor or numpy array.")
    else:
        B = len(imgs) if imgs.ndim == 4 else 1
        C = imgs.shape[-1]

    if (ncols is not None) and (B % ncols == 0):
        nrows = B // ncols
    else:
        nrows = 1
        ncols = B

    offset = 0 if not colorbar else 0.75
    fig = plt.figure(figsize=(ncols * (figsize + offset), nrows * figsize))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows)
    gs.update(wspace=0.025, hspace=0.025)

    if cmap is None:
        cmap = "gray" if C == 1 else None
    i = 0

    if title is None:
        title = [""]
    if len(title) != B:
        title += [""] * (B - len(title))

    with warnings.catch_warnings():

        def add_colorbar(im, axs, vmin, vmax):
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="3%", pad=0.1)

            arr = im.get_array()
            if vmin is None:
                vmin = float(np.nanmin(imgs))
            if vmax is None:
                vmax = float(np.nanmax(imgs))
            ticks = np.linspace(vmin, vmax, num=5)
            cbar = plt.colorbar(im, cax=cax, ticks=ticks)
            if colorbar_geq and (vmax is not None):
                # prefer the explicit vmax the user gave, otherwise use computed one
                vmax_label = vmax if vmax is not None else vmax
                ticks = cbar.get_ticks()
                if len(ticks) > 0:
                    tick_labels = [f"{t:.3g}" for t in ticks[:-1]] + [f"≥{vmax_label}"]
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(tick_labels)

        if nrows > 1 and ncols > 1:
            for r in range(nrows):
                for c in range(ncols):
                    axs = plt.subplot(gs[r, c])
                    im = axs.imshow(
                        imgs[i],
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        interpolation=interpolation,
                    )
                    axs.set_xticks([])
                    axs.set_yticks([])
                    if colorbar:
                        add_colorbar(im, axs, vmin, vmax)
                    if title:
                        axs.set_title(title[i], pad=3)
                    i += 1
        elif nrows > 1 or ncols > 1:
            for c in range(max(ncols, nrows)):
                axs = plt.subplot(gs[c])
                im = axs.imshow(
                    imgs[c],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    interpolation=interpolation,
                )
                axs.set_xticks([])
                axs.set_yticks([])
                if colorbar:
                    add_colorbar(im, axs, vmin, vmax)
                if title:
                    axs.set_title(title[c], pad=3)
        else:
            axs = plt.subplot(gs[0])
            im = axs.imshow(
                imgs.squeeze(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation=interpolation,
            )
            axs.set_xticks([])
            axs.set_yticks([])
            if colorbar:
                add_colorbar(im, axs, vmin, vmax)
            if title:
                axs.set_title(title[0], pad=3)

        if suptitle is not None:
            fig.suptitle(suptitle, y=1.03)
        if savename is not None:
            fig.savefig(savename, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            plt.show()

