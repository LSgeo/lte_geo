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
            "hr_line_spacing": kwargs.get("line_spacing", 1),
            "sample_spacing": kwargs.get("sample_spacing", 20),
            "heading": kwargs.get("heading", None),  # Default will be random
        }
        kwargs["model_dir"] = root_path
        self.scale = -1  # init params
        self.inp_size = -1
        super().__init__(**kwargs)

    def __len__(self):
        return self.len

    def _subsample(self, raster, ls):
        """Select points from raster according to line spacing"""
        # input_cell_size = 20 # Noddyverse cell size is 20 m
        ss = 1  # Sample all points along line

        xx, yy = np.meshgrid(
            np.arange(raster.shape[-1]),  # x, cols
            np.arange(raster.shape[-2]),  # y, rows
            indexing="xy",
        )
        xx = xx[::ss, ::ls]
        yy = yy[::ss, ::ls]
        z = raster.numpy()[:, ::ss, ::ls].squeeze()  # shape for gridding

        return xx, yy, z

    def _grid(self, x, y, z, scale, ls, cs_fac=4):
        """Min Curvature grid xyz at scale, with 1/5 cell size to ls ratio"""
        scale = 1
        scale = 2
        scale = 3
        scale = 4

        ls = 4
        ls *= scale
        w0 = s0 = 0  # Pixels to move in from boundary
        d = 200  # Limit to non-NAN extent
        # crop = int(self.inp_size // scale)

        w, e, s, n = np.array([w0, w0 + d, s0, s0 + d], dtype=np.float32)
        gridder = vd.ScipyGridder("cubic")
        gridder = gridder.fit((x, y), z)
        grid = gridder.grid(
            data_names="forward",
            coordinates=np.meshgrid(
                np.arange(w, e, step=ls / cs_fac),
                np.arange(s, n, step=ls / cs_fac),
            ),
        )
        grid = grid.get("forward").values.astype(np.float32)
        # grid = grid[:crop, :crop]

        return np.expand_dims(grid, 0)  # re-add channel dimension

    def _process(self, index):
        super()._process(index)

        hls = self.sp["hr_line_spacing"]
        lls = int(hls * self.scale)
        hr_x, hr_y, hr_z = self._subsample(self.data["gt_grid"], hls)
        lr_x, lr_y, lr_z = self._subsample(self.data["gt_grid"], lls)
        self.data["hr_grid"] = self._grid(hr_x, hr_y, hr_z, scale=1, ls=hls)
        self.data["lr_grid"] = self._grid(lr_x, lr_y, lr_z, scale=self.scale, ls=lls)

    def __getitem__(self, index):
        self._process(index)
        return self.data


@register("noddyverse_wrapper")
class NoddyverseWrapper(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        scale_min=1,
        scale_max=None,
        augment=False,
        sample_q=None,
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.dataset.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max or scale_min  # if not scale_max...
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        scale = torch.randint(low=self.scale_min, high=self.scale_max + 1, size=(1,))
        self.dataset.scale = scale

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
    These data include 1 nT Gaussian noise.

    https://doi.org/10.1190/geo2018-0156.1
    https://github.com/TomasNaprstek/Naprstek-Smith-Interpolation/tree/master/StandAlone
    """
    from pathlib import Path
    import colorcet as cc
    import matplotlib.pyplot as plt

    txt_file = next(Path(root).glob(data_txt_file))
    grid = np.loadtxt(txt_file, skiprows=1, dtype=np.float32)

    y = x = np.arange(start=2.5, stop=3000, step=5)
    grid = np.rot90(grid[:, 2].reshape(600, 600, 1))

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

    return grid
