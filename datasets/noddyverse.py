from pathlib import Path
import numpy as np
import torch
import verde as vd
from torch.utils.data import Dataset

from mlnoddy.datasets import NoddyDataset, Norm

from . import register
from ..utils import to_pixel_samples

rng = np.random.default_rng(21)


@register("noddyverse_dataset")
class HRLRNoddyverse(NoddyDataset):
    """
    Find a Noddyverse model         - super()._process
    Load the magnetic forward model - super()._process
    Normalise the GT forward model  - self.norm()
    Sample the forward model points - self._subsample()
    Grid the sampled points         - self._grid()
    Sample the grid points For LTE  - utils.to_pixel_samples()
    """

    def __init__(
        self,
        root_path=None,
        split_file=None,
        split_key=None,
        first_k=None,
        repeat=1,
        cache="none",
        **kwargs,
    ):
        self.sp = {
            "hr_line_spacing": kwargs.get("hr_line_spacing", 4),
            "sample_spacing": kwargs.get("sample_spacing", 1),
            "heading": kwargs.get("heading", "NS"),  # "EW" untested
            "noise": kwargs.get(
                "noise", {"gaussian": None, "geology": None, "levelling": None}
            ),
        }
        kwargs["model_dir"] = root_path
        self.scale = None  # init params
        self.inp_size = kwargs.get("input_size", None)  # set in super init?
        self.crop = None  # set after wrapper
        self.repeat = repeat
        super().__init__(**kwargs)

    def __len__(self):
        return self.len * self.repeat  # Repeat dataset like an epoch

    def _subsample(self, raster, ls):
        """Select points from raster according to line spacing"""
        # input_cell_size = 20 # Noddyverse cell size is 20 m
        ss = self.sp["sample_spacing"]  # Sample every n points along line

        x, y = np.meshgrid(
            np.arange(raster.shape[-1]),  # x, cols
            np.arange(raster.shape[-2]),  # y, rows
            indexing="xy",
        )
        x = x[::ss, ::ls]
        y = y[::ss, ::ls]
        vals = raster.numpy()[:, ::ss, ::ls].squeeze()  # shape for gridding
        # self.og_vals = vals
        # self.noise = np.zeros_like(vals)

        # Data are normalised to +-1 by this time.
        r = vals.max() - vals.min()  # 2  # config "rgb_range"
        if self.sp["noise"]["gaussian"] is not None:  # Sensor noise
            noise = (
                rng.normal(scale=0.25, size=vals.shape)
                * r
                * self.sp["noise"]["gaussian"]
            )
            # self.noise += noise
            vals += noise

        # if self.sp["noise"]["geology"] is not None:
        # Regional geology noise effect is accounted for in Noddyverse

        if self.sp["noise"]["levelling"] is not None:  # Line levelling
            for val_col in vals.T:  # noise_col, zip( ,self.noise.T)
                noise = rng.normal(scale=0.25) * r * self.sp["noise"]["levelling"]
                # noise_col += noise
                val_col += noise

        return x, y, vals

    def _grid(self, x, y, z, ls, cs_fac=4, d=180):
        """Min Curvature grid xyz at scale, with ls/cs_fac cell size.
        Params:
            d: adjustable crop factor, but 180 is best for noddyverse. 200 Max.
        """
        w, e, s, n = np.array([0, d, 0, d], dtype=np.float32)
        cs = ls / cs_fac  # Cell size is e.g. 1/4 line spacing
        gridder = vd.Cubic().fit(coordinates=(x, y), data=z)
        grid = gridder.grid(
            data_names="forward",
            coordinates=np.meshgrid(
                np.arange(w, e, step=cs),
                np.arange(s, n, step=cs),
                indexing="xy",
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

        hls = self.sp["hr_line_spacing"]  # normally set to 4
        lls = int(hls * self.scale)  # normally set in range(10)
        hr_x, hr_y, hr_z = self._subsample(self.data["gt_grid"], hls)
        lr_x, lr_y, lr_z = self._subsample(self.data["gt_grid"], lls)

        # I would like to norm prior gridding, but it breaks value ranges :(

        # Note - we use scale here as a factor describing how big HR is x LR.
        # I think this diverges from what my brain normally does.
        self.data["hr_grid"] = self.norm(self._grid(hr_x, hr_y, hr_z, ls=hls, d=d))
        self.data["lr_grid"] = self.norm(self._grid(lr_x, lr_y, lr_z, ls=lls, d=d))

        lr_extent = self.data["lr_grid"].shape[-1]
        # lr_extent = int((d / self.scale) * 4)  # cs_fac = 4

        if self.crop:
            # crop to inp_size, otherwise use full extent
            # We grid lr and hr at their full extent and crop the same patch
            # lr_extent is lr size _before_ cropping
            # lr size is self.inp_size _after_ cropping
            # hr size is self.inp_size * self.scale

            lr_e = int(
                torch.randint(low=0, high=lr_extent - self.inp_size + 1, size=(1,))
            )
            lr_n = int(
                torch.randint(low=0, high=lr_extent - self.inp_size + 1, size=(1,))
            )

            self.data["hr_grid"] = self._crop(
                self.data["hr_grid"], extent=(lr_e, lr_n), scale=self.scale
            )
            self.data["lr_grid"] = self._crop(
                self.data["lr_grid"], extent=(lr_e, lr_n), scale=1
            )

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
        crop=False,
        mode="lte",
        **kwargs,
    ):
        self.dataset = dataset
        self.dataset.inp_size = inp_size
        self.scale = scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max or scale_min  # if not scale_max...
        self.augment = augment
        self.sample_q = sample_q  # clip hr samples to same length
        self.crop = crop
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.dataset.crop = self.crop
        self.dataset.scale = torch.randint(
            low=self.scale_min,
            high=self.scale_max + 1,
            size=(1,),
        )  # int(self.scale)
        data = self.dataset[index]

        data["hr_grid"] = torch.from_numpy(data["hr_grid"]).to(torch.float32)
        data["lr_grid"] = torch.from_numpy(data["lr_grid"]).to(torch.float32)

        if "rdn" in self.mode:
            return {"hr": data["hr_grid"], "lr": data["lr_grid"], "gt": data["gt_grid"]}

        # data contains the hr and lr grids at their correct sizes.

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

        self.cache_path = Path("C:/Users/Public/scratch/temp") / ".cached_grid.npy"
        if self.cache_path.exists():
            print(f"Loading cached grid from {self.cache_path.absolute()}")
        else:
            Path("C:/Users/Public/scratch/temp").mkdir(exist_ok=True, parents=True)
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

    def __len__(self):
        return len(self.gt_patches) * self.repeat

    def __getitem__(self, index):
        idx = index % len(self.gt_patches)
        self.data = {}
        self.data["gt_grid"] = self.gt_patches[idx]

        return self.data


@register("real_dataset")
class HRLRReal(RealDataset):
    def __init__(
        self,
        root_path=None,
        **kwargs,
    ):
        self.sp = {
            "hr_line_spacing": kwargs.get("hr_line_spacing", 4),
            "sample_spacing": kwargs.get("sample_spacing", 1),
            "heading": kwargs.get("heading", "NS"),  # "EW" deprecated
        }
        kwargs["model_dir"] = root_path
        self.scale = None  # init params
        super().__init__(root_path, **kwargs)

    def _subsample(self, raster, ls):
        """Select points from raster according to line spacing"""
        ss = self.sp["sample_spacing"]  # Sample every n points along line

        x, y = np.meshgrid(
            np.arange(raster.shape[-1]),  # x, cols
            np.arange(raster.shape[-2]),  # y, rows
            indexing="xy",
        )
        x = x[::ss, ::ls]
        y = y[::ss, ::ls]
        vals = raster[:, ::ss, ::ls].squeeze()  # shape for gridding
        return x, y, vals

    def _grid(self, x, y, z, ls, cs_fac=4, d=180):
        """Min Curvature grid xyz at scale, with ls/cs_fac cell size.
        Params:
            d: adjustable crop factor, but 180 is best for noddyverse. 200 Max.
        """
        w, e, s, n = np.array([0, d, 0, d], dtype=np.float32)
        cs = ls / cs_fac  # Cell size is e.g. 1/4 line spacing
        gridder = vd.Cubic().fit(coordinates=(x, y), data=z)
        grid = gridder.grid(
            data_names="forward",
            coordinates=np.meshgrid(
                np.arange(w, e, step=cs),
                np.arange(s, n, step=cs),
                indexing="xy",
            ),
        )
        grid = grid.get("forward").values.astype(np.float32)

        return np.expand_dims(grid, 0)  # add channel dimension

    def process(self, d=180):
        # d is pixels in x, y to crop NAN from (if last row/col not sampled)
        hls = self.sp["hr_line_spacing"]  # normally set to 4
        lls = int(hls * self.scale)  # normally set in range(10)
        hr_x, hr_y, hr_z = self._subsample(self.data["gt_grid"], hls)
        lr_x, lr_y, lr_z = self._subsample(self.data["gt_grid"], lls)

        self.data["hr_grid"] = self.norm(self._grid(hr_x, hr_y, hr_z, ls=hls, d=d))
        self.data["lr_grid"] = self.norm(self._grid(lr_x, lr_y, lr_z, ls=lls, d=d))


@register("real_wrapper")
class RealWrapper(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        augmentations: dict = {},
        scale_min=2,
        scale_max=None,
        sample_q=None,
    ):
        self.dataset = dataset
        self.dataset.inp_size = inp_size
        self.scale = scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max or scale_min  # if not scale_max...
        self.aug = augmentations
        self.sample_q = sample_q  # clip hr samples to same length
        self.aug_count = 1 + sum(augmentations.values())  # original data included

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

    def __len__(self):
        return len(self.dataset) * self.aug_count

    def __getitem__(self, index):
        self.dataset.scale = torch.randint(
            low=self.scale_min,
            high=self.scale_max + 1,
            size=(1,),
        )  # int(self.scale)

        data = self.dataset[index]

        if self.aug.get("sample"):
            self._preprocess_augment(data)
        self.dataset.process()  # d = cfg.gt_patch_size
        if self.aug is not None:
            self._augment(data)

        data["gt_grid"] = torch.from_numpy(data["gt_grid"].copy()).to(torch.float32)
        data["hr_grid"] = torch.from_numpy(data["hr_grid"].copy()).to(torch.float32)
        data["lr_grid"] = torch.from_numpy(data["lr_grid"].copy()).to(torch.float32)

        hr_coord, hr_val = to_pixel_samples(data["hr_grid"].contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_val = hr_val[sample_lst]

        hr_cell = torch.ones_like(hr_coord)
        hr_cell[:, 0] *= 2 / data["hr_grid"].shape[-2]
        hr_cell[:, 1] *= 2 / data["hr_grid"].shape[-1]

        return {
            "inp": data["lr_grid"],
            "coord": hr_coord,
            "cell": hr_cell,
            "gt": hr_val,
        }
