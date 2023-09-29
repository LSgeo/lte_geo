from pathlib import Path
import numpy as np
import torch
import verde as vd
from torch.utils.data import Dataset

from mlnoddy.datasets import NoddyDataset, Norm

from . import register
from ..utils import to_pixel_samples

rng = np.random.default_rng(21)


def subsample(raster, ls, sp):
    """Select points from (raster) according to line spacing (ls) and other
    survey parameters (sp).
    """
    # Noddyverse cell size is 20 m
    inp_cs = sp["input_cell_size"]
    ss = int(sp["sample_spacing"] / inp_cs)  # Sample every n points N-S along line
    ls = int(ls / inp_cs)

    x, y = np.meshgrid(
        np.arange(raster.shape[-1]),
        np.arange(raster.shape[-2]),
        indexing="xy",
    )
    x1 = x[::ss, ::ls]
    x2 = x[::ss, -1]
    x = np.concatenate((x1, x2[:, None]), axis=1)

    y1 = y[::ss, ::ls]
    y2 = y[::ss, -1]
    y = np.concatenate((y1, y2[:, None]), axis=1)

    v1 = raster[:, ::ss, ::ls].squeeze()  # shape for gridding
    v2 = raster[:, ::ss, -1].squeeze()
    vals = np.concatenate((v1, v2[:, None]), axis=1)

    return x, y, vals


def add_noise(vals, sp):
    # Data are normalised to +-1 by this time.
    r = vals.max() - vals.min()  # 2  # config "rgb_range"

    # if self.sp["noise"]["geology"] is not None:
    # Regional geology noise effect is accounted for in Noddyverse

    if sp["noise"]["gaussian"] is not None:  # Sensor noise
        noise = rng.normal(scale=0.25, size=vals.shape) * r * sp["noise"]["gaussian"]
        # self.noise += noise
        vals += noise

    if sp["noise"]["levelling"] is not None:  # Line levelling
        for val_col in vals.T:  # noise_col, zip( ,self.noise.T)
            noise = rng.normal(scale=0.25) * r * sp["noise"]["levelling"]
            # noise_col += noise
            val_col += noise

    return vals


def grid(x, y, vals, ls, cs_fac=5, d=4000, inp_cs=20):
    """Min Curvature grid xyz, with ls/cs_fac cell size.
    Params:
        x, y, vals: subsampled coord/val points
        ls: line spacing of data
        cs_fac: 4 or 5 as rule of thumb
        d: adjustable crop factor, spatial units to match ls

    """
    w, e, s, n = np.array([0, d, 0, d], dtype=np.float32)
    cs = ls / cs_fac  # Cell size is e.g. 1/4 line spacing
    gridder = vd.Cubic().fit(coordinates=(x * inp_cs, y * inp_cs), data=vals)
    coordinates = vd.grid_coordinates((w, e, s, n), spacing=cs, pixel_register=True)
    grid = gridder.grid(data_names="forward", coordinates=coordinates)
    grid = grid.get("forward").values.astype(np.float32)

    if False:
        import matplotlib.pyplot as plt

        plt.imshow(grid)

    return np.expand_dims(grid, 0)  # add channel dimension


# def crop(inp_size, scale, lr_grid, hr_grid):
#     """crop to inp_size, otherwise use full extent
#     We grid lr and hr at their full extent and crop the same patch
#     """
#     lr_extent = lr_grid.shape[-1]
#     lr_e = int(torch.randint(low=0, high=lr_extent - inp_size + 1, size=(1,)))
#     lr_n = int(torch.randint(low=0, high=lr_extent - inp_size + 1, size=(1,)))
#     hr_e = lr_e * scale
#     hr_n = lr_n * scale

#     lr_grid = lr_grid[:, lr_e : lr_e + inp_size, lr_n : lr_n + inp_size]
#     hr_grid = hr_grid[:, hr_e : hr_e + inp_size * scale, hr_n : hr_n + inp_size * scale]

#     return lr_grid, hr_grid


@register("noddyverse_dataset")
class HRLRNoddyverse(NoddyDataset):
    """
    Find a Noddyverse model         - super()._process
    Load the magnetic forward model - super()._process
    Normalise the GT forward model  - self.norm()
    Sample the forward model points - subsample()
    Grid the sampled points         - self._grid()
    Sample the grid points For LTE  - utils.to_pixel_samples()
    """

    def __init__(
        self,
        root_path=None,
        repeat=1,
        **kwargs,
    ):
        kwargs["model_dir"] = root_path
        self.scale = None  # Set in Wrapper
        # self.crop = None  # Set in Wrapper
        self.repeat = repeat
        self.sp = {**kwargs}
        super().__init__(**kwargs)

    def __len__(self):
        return self.len * self.repeat  # Repeat dataset like an epoch

    def __getitem__(self, index):
        idx = index % self.len  # for repeating
        self.process(idx)
        return self.data

    def process(self, index):
        # d is pixels in x, y to crop NAN from (if last row/col not sampled)
        super()._process(index)

        lls = self.sp["lr_line_spacing"]  # This should be 400
        hls = lls / self.scale  # scale is 2, 4 or 10. So hls is 200, 100 or 40

        lr_x, lr_y, lr_val = subsample(self.data["gt_grid"], lls, self.sp)
        hr_x, hr_y, hr_val = subsample(self.data["gt_grid"], hls, self.sp)

        if self.sp.get("noise"):
            lr_val = add_noise(lr_val, self.sp)
            hr_val = add_noise(hr_val, self.sp)

        self.data["lr_grid"] = grid(lr_x, lr_y, lr_val, ls=lls)
        self.data["hr_grid"] = grid(hr_x, hr_y, hr_val, ls=hls)

        self.data["lr_grid"] = self.norm(self.data["lr_grid"])
        self.data["hr_grid"] = self.norm(self.data["hr_grid"])

        # if self.crop:
        #     self.data["lr_grid"], self.data["hr_grid"] = crop(
        #         self.inp_size, self.scale, self.data["lr_grid"], self.data["hr_grid"]
        #     )


@register("noddyverse_wrapper")
class NoddyverseWrapper(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        augmentations: dict = {},
        scales=None,
        sample_q=None,
        mode="lte",
        **kwargs,
    ):
        self.dataset = dataset
        self.dataset.inp_size = inp_size
        self.override_scale = None
        self.scales = scales
        self.aug = augmentations
        self.sample_q = sample_q  # clip hr samples to same length
        self.mode = mode
        self.aug_count = 1 + sum(augmentations.values())  # original data included

    def __len__(self):
        return len(self.dataset) * self.aug_count

    def __getitem__(self, index):
        self.dataset.scale = rng.choice(self.scales)
        if self.override_scale:
            self.dataset.scale = self.override_scale
            self.override_scale = None

        data = self.dataset[index]

        data["hr_grid"] = torch.from_numpy(data["hr_grid"]).to(torch.float32)
        data["lr_grid"] = torch.from_numpy(data["lr_grid"]).to(torch.float32)

        if "rdn" in self.mode:
            return {"hr": data["hr_grid"], "lr": data["lr_grid"], "gt": data["gt_grid"]}

        hr_coord, hr_val = to_pixel_samples(data["hr_grid"].contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_val = hr_val[sample_lst]

        hr_cell = torch.ones_like(hr_coord)
        hr_cell[:, 0] *= 2 / data["hr_grid"].shape[-2]
        hr_cell[:, 1] *= 2 / data["hr_grid"].shape[-1]

        if "lte" in self.mode:
            return {
                "inp": data["lr_grid"],
                "coord": hr_coord,
                "cell": hr_cell,
                "gt": hr_val,
            }


class RealDataset(Dataset):
    """Load a real survey tif and extract a patch to use as GT data.
    GT data will be subsampled and gridded in noddyvesrse.HRLRReal() to
    create HR and LR data.
    """

    def __init__(self, root_path, norm, **kwargs):
        super().__init__()
        if norm is not None:
            self.norm = Norm(
                clip_min=norm[0], clip_max=norm[1], out_vals=(0, 1)
            ).min_max_clip
            self.unorm = Norm(
                clip_min=norm[0], clip_max=norm[1], out_vals=(0, 1)
            ).inverse_mmc
        else:
            self.norm = Norm(out_vals=(0, 1)).per_sample_norm
            self.unorm = Norm(out_vals=(0, 1)).inverse_mmc

        self.root_path = Path(root_path)
        self.repeat = kwargs.get("repeat", 1)
        self.gt_patches = np.load(self.root_path, mmap_mode="r")
        self.gt_patch_size = kwargs.get("gt_patch_size")


@register("real_dataset")
class HRLRReal(RealDataset):
    def __init__(
        self,
        root_path=None,
        repeat=1,
        **kwargs,
    ):
        kwargs["model_dir"] = root_path
        self.scale = None  # init params
        # self.crop = None  # set in wrapper
        self.repeat = repeat
        self.sp = {**kwargs}
        super().__init__(root_path, **kwargs)

    def __len__(self):
        return len(self.gt_patches) * self.repeat

    def __getitem__(self, index):
        idx = index % len(self.gt_patches)
        self.data = {}
        self.data["gt_grid"] = torch.from_numpy(self.gt_patches[idx]).to(torch.float32)
        return self.data

    def process(self):
        lls = self.sp["lr_line_spacing"]  # This should be 400
        hls = lls / self.scale  # scale is 2, 4 or 10. So hls is 200, 100 or 40

        lr_x, lr_y, lr_val = subsample(self.data["gt_grid"], lls, self.sp)
        hr_x, hr_y, hr_val = subsample(self.data["gt_grid"], hls, self.sp)

        if self.sp.get("noise"):
            lr_val = add_noise(lr_val, self.sp)
            hr_val = add_noise(hr_val, self.sp)

        self.data["lr_grid"] = grid(lr_x, lr_y, lr_val, ls=lls)
        self.data["hr_grid"] = grid(hr_x, hr_y, hr_val, ls=hls)

        self.data["lr_grid"] = self.norm(self.data["lr_grid"])
        self.data["hr_grid"] = self.norm(self.data["hr_grid"])

        # if self.crop:
        #     self.data["lr_grid"], self.data["hr_grid"] = crop(
        #         self.inp_size, self.scale, self.data["lr_grid"], self.data["hr_grid"]
        #     )


@register("real_wrapper")
class RealWrapper(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        augmentations: dict = {},
        scales=None,
        sample_q=None,
        mode="lte",
        **kwargs,
    ):
        self.dataset = dataset
        self.dataset.inp_size = inp_size
        self.scales = scales
        self.override_scale = None
        self.aug = augmentations
        self.sample_q = sample_q  # clip hr samples to same length
        self.mode = mode
        self.aug_count = 1 + sum(augmentations.values())  # original data included

    def __len__(self):
        return len(self.dataset) * self.aug_count

    def __getitem__(self, index):
        self.dataset.scale = rng.choice(self.scales)
        if self.override_scale:
            self.dataset.scale = self.override_scale
            self.override_scale = None

        data = self.dataset[index]

        if self.aug.get("sample"):
            self._preprocess_augment(data)

        self.dataset.process()  # d = cfg.gt_patch_size

        if self.aug is not None:
            self._augment(data)

        # data["gt_grid"] = torch.from_numpy(data["gt_grid"].copy()).to(torch.float32)
        data["hr_grid"] = torch.from_numpy(data["hr_grid"].copy()).to(torch.float32)
        data["lr_grid"] = torch.from_numpy(data["lr_grid"].copy()).to(torch.float32)

        if "rdn" in self.mode:
            return {"hr": data["hr_grid"], "lr": data["lr_grid"], "gt": data["gt_grid"]}

        hr_coord, hr_val = to_pixel_samples(data["hr_grid"].contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_val = hr_val[sample_lst]

        hr_cell = torch.ones_like(hr_coord)
        hr_cell[:, 0] *= 2 / data["hr_grid"].shape[-2]
        hr_cell[:, 1] *= 2 / data["hr_grid"].shape[-1]

        if "lte" in self.mode:
            return {
                "inp": data["lr_grid"],
                "coord": hr_coord,
                "cell": hr_cell,
                "gt": hr_val,
            }

    def _preprocess_augment(self, data):
        """Rotate GT, pre-subsampling / gridding."""
        self.aug_count += 1
        if torch.rand(1) < 0.5:
            data["gt_grid"] = np.rot90(data["gt_grid"], axes=(1, 2))

    def _augment(self, data):
        """Augment data with flips and rotations"""
        if self.aug.get("flip"):
            self.aug_count += 2
            if torch.rand(1) < 0.5:
                data["gt_grid"] = np.fliplr(data["gt_grid"])
                data["hr_grid"] = np.fliplr(data["hr_grid"])
                data["lr_grid"] = np.fliplr(data["lr_grid"])
            if torch.rand(1) < 0.5:
                data["gt_grid"] = np.flipud(data["gt_grid"])
                data["hr_grid"] = np.flipud(data["hr_grid"])
                data["lr_grid"] = np.flipud(data["lr_grid"])
        if self.aug.get("rotate"):
            self.aug_count += 1
            if torch.rand(1) < 0.5:
                data["gt_grid"] = np.rot90(data["gt_grid"], axes=(1, 2))
                data["hr_grid"] = np.rot90(data["hr_grid"], axes=(1, 2))
                data["lr_grid"] = np.rot90(data["lr_grid"], axes=(1, 2))

        # _DEBUG = False
        # if _DEBUG:
        #     import matplotlib.pyplot as plt
        #     import colorcet as cc
        #     from torchvision.transforms.functional import resize
        #     from torchvision.transforms import InterpolationMode

        #     fig, [
        #         [axgt, axhr, axlr],
        #         [axoff, axdgh, axdlr],
        #     ] = plt.subplots(2, 3)
        #     plt.suptitle(f"{self.scale}x gridding debug visualisations_id-{index}")

        #     axgt.set_title("gt")
        #     axgt.imshow(self.data["gt_grid"][:, :180, :180].squeeze())
        #     axhr.set_title("hr")
        #     axhr.imshow(self.data["hr_grid"].squeeze())
        #     axlr.set_title("lr")
        #     axlr.imshow(self.data["lr_grid"].squeeze())
        #     axoff.set_axis_off()
        #     axdgh.set_title("gt - hr")
        #     dgh = axdgh.imshow(
        #         (self.data["gt_grid"][:, :180, :180] - self.data["hr_grid"]).squeeze(),
        #         cmap=cc.cm.CET_D1,
        #     )
        #     plt.colorbar(dgh, ax=axdgh, location="bottom")
        #     axdlr.set_title("resize_gt - lr")
        #     dlr = axdlr.imshow(
        #         (
        #             resize(
        #                 self.data["gt_grid"][:, :180, :180],
        #                 self.data["lr_grid"].shape[1:],
        #                 InterpolationMode.BICUBIC,
        #             )
        #             - self.data["lr_grid"]
        #         ).squeeze(),
        #         cmap=cc.cm.CET_D1,
        #     )
        #     plt.colorbar(dlr, ax=axdlr, location="bottom")

        # if _DEBUG:
        #     plt.close()


def load_naprstek_synthetic(
    root="D:/luke/Noddy_data/test",
    data_txt_file="Naprstek_BaseModel1-AllValues-1nTNoise.txt",
    normalise=False,
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
    from mlnoddy.datasets import Norm

    txt_file = next(Path(root).glob(data_txt_file))
    grid = np.loadtxt(txt_file, skiprows=1, dtype=np.float32)
    grid = torch.from_numpy(grid[:, 2]).reshape(600, 600, 1)

    # Visualise as per Naprstek and Smith
    # import colorcet as cc
    # import matplotlib.pyplot as plt
    # y = x = np.arange(start=2.5, stop=3000, step=5)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.title("Total field (nT)")
    # plt.imshow(
    #     grid.rot90().permute(2, 0, 1).squeeze(),
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

    if normalise:
        norm = Norm(clip_min=-5000, clip_max=5000).min_max_clip
        return norm(grid.rot90().permute(2, 0, 1))
    else:
        return grid.rot90().permute(2, 0, 1)


class LargeRasterData:
    """Process large data extents to stacked patches for training/etc.
    e.g. The 30 GB state map, masked to <80 m, needs to be handled.
    We are going to patch extract it at 1 or 2 offsets, and save the
    finished patch stack to disk, for memmap loading.

    - Load the full grid
    - Run patch_extract on it for a few offsets
    - Save the resulting tensors to numpy array npy.

    """

    def __init__(self, file_path, nan_val=-99_999):
        self.file_path = Path(file_path)
        self.nan_val = nan_val

        self.cache_path = Path("temp") / ".cached_grid.npy"
        if self.cache_path.exists():
            print(f"Loading cached grid from {self.cache_path.absolute()}")
        else:
            Path("temp").mkdir(exist_ok=True, parents=True)
            print(f"Caching to {self.cache_path.absolute()}")
            if self.file_path.suffix == ".tif":
                import tifffile

                np.save(self.cache_path, tifffile.imread(self.file_path))

            elif self.file_path.suffix == ".ers":
                import rasterio

                with rasterio.open(self.file_path) as src:
                    if not self.nan_val == src.nodata:
                        raise ValueError(
                            f"Specified NaN value {self.nan_val} does not match ers metadata {src.nodata}"
                        )
                    np.save(self.cache_path, src.read(1))

        self.grid = np.load(self.cache_path, mmap_mode="c")
        self.grid = torch.from_numpy(self.grid)
        self.grid[self.grid == self.nan_val] = torch.nan

    def patch_extract(self, patch_size: int, offset: int = 0):
        """Extract a patch from the full gt grid.
        - Pad the grid to make an integer number of tiles across the extent
        - Unfold the grid into tiles, which may contain nan values
        - Drop any nan values, either from the original grid or padding

        offset: offset number of cells to pad to make patches unique
        e.g. [0, 51, 103, 157]
        """

        s = patch_size
        off_pad = offset
        self.off_pad = off_pad

        # Pad to make integer number of tiles across extent.
        pad_grid = torch.nn.functional.pad(
            self.grid,
            (
                off_pad,
                (s - (self.grid.shape[1] % s)),
                off_pad,
                (s - (self.grid.shape[0] % s)),
            ),
            mode="constant",
            value=torch.nan,
        )
        tiles_per_row = pad_grid.shape[1] // s
        tiles_per_column = pad_grid.shape[0] // s

        patches = pad_grid.unfold(0, s, s).unfold(1, s, s)
        patches = patches.contiguous().view(tiles_per_row * tiles_per_column, -1, s, s)

        # https://stackoverflow.com/a/64594975/10615407
        shape = patches.shape
        patches_reshaped = patches.reshape(shape[0], -1)
        patches_reshaped = patches_reshaped[~torch.any(patches_reshaped.isnan(), dim=1)]

        self.patches = patches_reshaped.reshape(
            patches_reshaped.shape[0], *shape[1:]
        ).numpy()

    def save_npy(self, out_path=None):
        if out_path is None:
            out_path = self.file_path.with_name(
                f"{self.file_path.stem}_offset_{self.off_pad}.npy"
            )
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.patches)
        print(f"Saved patches with shape {self.patches.shape} to {out_path.absolute()}")

    def cleanup(self):
        self.grid = None
        self.cache_path.unlink()

    def _eg():
        """What I ran to generate the data I used"""
        # from datasets.noddyverse import LargeRasterData
        trainingdata = LargeRasterData(
            file_path=Path(
                "C:/Users/Public/scratch/WA_20m_Mag_Merge_v1_2020/State Map surveys/processing/WA_MAG_20m_MGA2020_sub100m_train_2.tif"
            )
        )
        for offset in [0, 53, 107, 150]:
            trainingdata.patch_extract(patch_size=200, offset=offset)
            trainingdata.save_npy()
        trainingdata.cleanup()

        # Finally, merge all the created .npy files into one big one:
        stacked = []
        for arr in Path(
            "C:/Users/Public/scratch/WA_20m_Mag_Merge_v1_2020/State Map surveys/processing"
        ).glob("*2_offset_*.npy"):
            print("Stacking", arr.name)
            stacked.append(np.load(arr))

        stacked = np.concatenate(stacked, axis=0)
        np.save("C:/Users/Public/scratch/sub80m_patch_stack_train.npy", stacked)
