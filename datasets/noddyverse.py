import numpy as np
import torch
import verde as vd
from torch.utils.data import Dataset

from mlnoddy.datasets import NoddyDataset

from datasets import register
from utils import to_pixel_samples


@register("noddyverse_dataset")
class HRLRNoddyverse(NoddyDataset):
    """
    Find a Noddyverse model         - super()._process
    Load the magnetic forward model - super()._process
    Sample the forward model points - self._subsample()
    Grid the sampled points         - self._grid()
    Sample the grid points For LTE  - utils.to_pixel_samples()
    """

    def __init__(
        self,
        root_path,
        split_file=None,
        split_key=None,
        first_k=None,
        repeat=1,
        cache="none",
        **kwargs,
    ):
        self.sp = {
            "hr_line_spacing": kwargs.get("hr_line_spacing", 1),
            "sample_spacing": kwargs.get("sample_spacing", 20),
            "heading": kwargs.get("heading", None),  # Default will be random
        }
        kwargs["model_dir"] = root_path
        self.scale = None  # init params
        self.inp_size = None
        self.is_val = None # set after wrapper
        super().__init__(**kwargs)

    def __len__(self):
        return self.len

    def _subsample(self, raster, ls):
        """Select points from raster according to line spacing"""
        # input_cell_size = 20 # Noddyverse cell size is 20 m
        ss = 1  # Sample all (every 1) points along line

        xx, yy = np.meshgrid(
            np.arange(raster.shape[-1]),  # x, cols
            np.arange(raster.shape[-2]),  # y, rows
            indexing="xy",
        )
        xx = xx[::ss, ::ls]
        yy = yy[::ss, ::ls]
        z = raster.numpy()[:, ::ss, ::ls].squeeze()  # shape for gridding

        return xx, yy, z

    def _grid(self, x, y, z, ls, cs_fac=4, d=180):
        """Min Curvature grid xyz at scale, with ls/cs_fac cell size.
        Params:
            d: adjustable crop factor, but 180 is best for noddyverse. 200 Max.
        """
        w, e, s, n = np.array([0, d, 0, d], dtype=np.float32)
        cs = ls / cs_fac  # Cell size is e.g. 1/4 line spacing
        gridder = vd.ScipyGridder("cubic")
        gridder = gridder.fit(coordinates=(x, y), data=z)
        grid = gridder.grid(
            data_names="forward",
            coordinates=np.meshgrid(
                np.arange(w, e, step=cs),
                np.arange(s, n, step=cs),
            ),
        )
        grid = grid.get("forward").values.astype(np.float32)

        # w_grd = self.inp_size * scale.item()

        # w_lr = self.inp_size  # No matter what happens, lr is same size eg.48x48
        # w_hr = self.inp_size * self.scale.item()  # HR is e.g. 480*480 (10x)
        # # Max usable data is 720x720, which is around 15 times scale (cs_fac=4)
        # w_grd = w_hr if scale == 1 else w_lr

        # ##
        # # We want to grid a region that gives d size.
        # d = int(cs * self.inp_size)

        # # w0 = s0 = 0  # Pixels to move in from boundary
        # # d = 180  # Nice factors [1,10,] except 7 (and 8 if cs_fac==5).

        # # # Currently grid full extent and crop. May be better to direct grid to size.
        # # if scale == 1:
        # #     grid = grid[x0:x0+w_hr, y0:y0+w_hr]
        # # else:
        # #     grid = grid[x0:x0+w_lr, y0:y0+w_lr]

        # print(float(scale), grid.shape, np.isnan(grid).any())
        # grid = grid[:crop, :crop]

        return np.expand_dims(grid, 0)  # add channel dimension

    def _crop(self, grid, extent, scale):
        if self.is_val:
            return grid
        else:
            lr_e = extent[0] * scale
            lr_n = extent[1] * scale
            return grid[
                :,
                lr_e : lr_e + scale * self.inp_size,
                lr_n : lr_n + scale * self.inp_size,
            ]

    def _process(self, index, d=180):
        # d is pixels in x, y to crop NAN from (if last row/col not sampled)
        super()._process(index)

        hls = self.sp["hr_line_spacing"]
        lls = int(hls * self.scale)
        hr_x, hr_y, hr_z = self._subsample(self.data["gt_grid"], hls)
        lr_x, lr_y, lr_z = self._subsample(self.data["gt_grid"], lls)

        # Note - we use scale here as a factor describing how big HR is x LR.
        # I think this diverges from what my brain normally does.
        self.data["hr_grid"] = self._grid(hr_x, hr_y, hr_z, ls=hls, d=d)
        self.data["lr_grid"] = self._grid(lr_x, lr_y, lr_z, ls=lls, d=d)

        # We grid lr and hr at their full extent and crop the same patch
        # hr size is self.inp_size * self.scale
        # self.inp_size is lr size _after_ cropping
        # lr_extent is lr grid size _before_ cropping

        # lr_extent = int((d / self.scale) * 4)  # cs_fac = 4
        lr_extent = self.data["lr_grid"].shape[-1]
        lr_e = int(torch.randint(low=0, high=lr_extent - self.inp_size + 1, size=(1,)))
        lr_n = int(torch.randint(low=0, high=lr_extent - self.inp_size + 1, size=(1,)))

        self.data["hr_grid"] = self._crop(
            self.data["hr_grid"], extent=(lr_e, lr_n), scale=self.scale
        )
        self.data["lr_grid"] = self._crop(
            self.data["lr_grid"], extent=(lr_e, lr_n), scale=1
        )

    def __getitem__(self, index):
        self._process(index)
        return self.data


@register("noddyverse_wrapper")
class NoddyverseWrapper(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        scale_min=2,
        scale_max=None,
        augment=False,
        sample_q=None,
        is_val=False,
    ):
        self.dataset = dataset
        self.dataset.inp_size = inp_size
        self.scale = scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max or scale_min  # if not scale_max...
        self.augment = augment
        self.sample_q = sample_q  # clip hr samples to same length
        self.is_val = is_val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.dataset.is_val = self.is_val
        self.dataset.scale = int(self.scale)
        data = self.dataset[index]

        data["hr_grid"] = torch.from_numpy(data["hr_grid"]).to(torch.float32)
        data["lr_grid"] = torch.from_numpy(data["lr_grid"]).to(torch.float32)
        # data contains the hr and lr grids at their correct sizes.

        hr_coord, hr_val = to_pixel_samples(data["hr_grid"].contiguous())
        hr_cell = torch.ones_like(hr_coord)
        hr_cell[:, 0] *= 2 / data["hr_grid"].shape[-2]
        hr_cell[:, 1] *= 2 / data["hr_grid"].shape[-1]

        return {
            "inp": data["lr_grid"],
            "coord": hr_coord,
            "cell": hr_cell,
            "gt": hr_val,
        }


def load_naprstek_synthetic(
    root="D:/luke/Noddy_data/test",
    data_txt_file="Naprstek_BaseModel1-AllValues-1nTNoise.txt",
):
    """Parse synthetic data from Naprstek's GitHub.
    Includes a plot function to mimic the presentation done in their paper.

    ~Un~fortunately, I cannot mimic the exact rainbow stretch used
    by Oasis Montaj.

    Note the grid is 600x600, nominally 5 m cell size, for 3x3 km extent.
    These data include 1 nT Gaussian noise added to the forward model.

    https://doi.org/10.1190/geo2018-0156.1
    https://github.com/TomasNaprstek/Naprstek-Smith-Interpolation/tree/master/StandAlone
    """
    from pathlib import Path

    txt_file = next(Path(root).glob(data_txt_file))
    grid = np.loadtxt(txt_file, skiprows=1, dtype=np.float32)

    y = x = np.arange(start=2.5, stop=3000, step=5)
    grid = torch.from_numpy(grid[:, 2]).reshape(600, 600, 1)

    # import colorcet as cc
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.title("Total field (nT)")
    # plt.imshow(
    #     grid,
    #     origin="upper",
    #     cmap=cc.cm.CET_R1,
    #     extent=(x.min(), x.max(), y.min(), y.max()),
    #     vmin=-20,
    #     vmax=40,
    # )
    # plt.colorbar(location="top")
    # plt.xlabel("Model x position (m)")
    # plt.ylabel("Model y position (m)")
    # ax.xaxis.set_major_locator(plt.MultipleLocator(250))
    # plt.grid(which="both", axis="x", c='k')

    return grid.rot90().permute(2, 0, 1)
