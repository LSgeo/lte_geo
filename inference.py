from functools import partial
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import datasets as dsets
import models
import utils
from test import reshape, batched_predict
from mlnoddy.datasets import parse_geophysics, Norm

import rasterio
import tifffile
from datasets.noddyverse import HRLRNoddyverse, NoddyverseWrapper
from datasets.noddyverse import load_naprstek_synthetic as load_naprstek


def load_model(config, device="cuda"):
    model_dir = Path(cfg["model_dir"])
    model_name = cfg["model_name"]
    model_paths = list(model_dir.glob(f"**/*{model_name}*best.pth"))
    if len(model_paths) != 1:
        raise FileNotFoundError(
            f"No unique model found in {model_dir} for *{model_name}*best.pth."
        )

    model_spec = torch.load(model_paths[0], map_location=device)["model"]
    model = models.make(model_spec, load_sd=True).to(device)
    return model, model_name, model_paths


def main():
    model, model_name, model_paths = load_model(cfg)

    # Define Data ###
    spec = cfg["test_dataset"]
    dataset = dsets.make(spec["dataset"])
    dataset = dsets.make(spec["wrapper"], args={"dataset": dataset})
    # dataset.crop = spec["wrapper"]["args"]["crop"]
    # ^ this should now be handled in .make()

    if cfg["limit_to_plots"]:
        dataset = Subset(dataset, cfg["plot_samples"])

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        num_workers=cfg.get("num_workers"),
        persistent_workers=bool(cfg.get("num_workers")),
        pin_memory=True,
    )

    # Pack Options ###
    opts = dict(
        model_name=model_name,
        model_path=model_paths[0],
        save_path=Path(cfg["inference_output_dir"] or f"inference/{model_name}"),
        rgb_range=cfg["rgb_range"],
        shave_factor=3,  # pixels to shave (edges may include NaN)
        ids=cfg["plot_samples"],  # Sample IDs
        mag=cfg["test_dataset"]["dataset"]["args"]["load_magnetics"],
        grv=cfg["test_dataset"]["dataset"]["args"]["load_gravity"],
    )

    scale_min = spec["wrapper"]["args"]["scale_min"]
    scale_max = spec["wrapper"]["args"]["scale_max"]

    print(
        f"\nModel: {opts['model_path'].absolute()}\n"
        f"Saving to: {opts['save_path'].absolute()}"
    )

    # Do Inference ###
    results_dict = {}
    custom_results_dict = {}
    pbar_m = tqdm(range(scale_min, scale_max + 1))
    for scale in pbar_m:
        pbar_m.set_description(f"{scale}x scale")
        opts["shave"] = scale * opts["shave_factor"]

        # lazy way to reset custom grid opts
        opts["ids"] = cfg["plot_samples"]
        opts["gt"] = None
        opts["set"] = "test"

        dataset.scale = scale
        if cfg["limit_to_plots"]:
            dataset.dataset.scale = scale

        results = eval(model, scale, loader, opts)
        results_dict[f"{scale}x"] = results
        pbar_m.write(f"{scale}x scale - Mean:")
        pbar_m.write(
            ", ".join(
                f"{metric_name}: {metric_value:.4f}"
                for metric_name, metric_value in results.items()
            )
        )

        if cfg["custom_grids"]:
            opts["set"] = "Custom"
            results = test_custom_data(model, scale, opts)
            custom_results_dict[f"{scale}x - Custom"] = results
            pbar_m.write(f"Naprstek {scale}x scale - Mean:")
            pbar_m.write(
                ", ".join(
                    f"{metric_name}: {metric_value:.4f}"
                    for metric_name, metric_value in results.items()
                )
            )
    opts["set"] = "test"
    plt_results(results_dict, opts)
    opts["set"] = "custom"
    plt_results(custom_results_dict, opts)

    return results_dict


def eval(model, scale, loader, opts):
    model.eval()

    l1_fn = torch.nn.L1Loss()
    l1_avg = utils.Averager()
    psnr_fn = partial(
        utils.calc_psnr,
        dataset="noddyverse",
        scale=scale,
        rgb_range=cfg["rgb_range"],
        shave=opts["shave"],
    )
    psnr_avg = utils.Averager()

    pbar = tqdm(loader, leave=False, desc="Test")
    for i, batch in enumerate(pbar):
        inp = batch["inp"].to("cuda", non_blocking=True)
        coord = batch["coord"].to("cuda", non_blocking=True)
        cell = batch["cell"].to("cuda", non_blocking=True)
        batch["gt"] = batch["gt"].to("cuda", non_blocking=True)

        with torch.no_grad():
            if cfg["eval_bsize"]:
                pred = batched_predict(model, inp, coord, cell, cfg["eval_bsize"])
            else:
                pred = model(inp, coord, cell)

        pred, batch = reshape(batch, h_pad=0, w_pad=0, coord=coord, pred=pred)

        l1 = l1_fn(pred, batch["gt"])
        l1_avg.add(l1.item(), inp.shape[0])
        psnr = psnr_fn(pred, batch["gt"])
        psnr_avg.add(psnr.item(), inp.shape[0])

        pbar.set_description(
            f"Mean: L1: {l1_avg.item():.4f}, PSNR: {psnr_avg.item():.4f}"
        )

        if cfg["limit_to_plots"]:
            lr = batch["inp"].detach().cpu().squeeze().numpy()
            hr = batch["gt"].detach().cpu().squeeze().numpy()
            sr = pred.detach().cpu().squeeze().numpy()

            suffix = opts["ids"][i]
            if opts["mag"]:
                geo_d = "mag"
            elif opts["grv"]:
                geo_d = "grv"
            else:
                raise NotImplementedError
            opts["geo_d"] = geo_d

            if opts.get("gt") is not None:
                gt = opts["gt"].squeeze()
            else:
                gt_list = Path(cfg["test_dataset"]["dataset"]["args"]["root_path"])
                gt_list = [Path(str(p)[:-3]) for p in gt_list.glob(f"**/*.{geo_d}.gz")]
                gt = next(
                    parse_geophysics(
                        gt_list[opts["ids"][i]],
                        mag=opts["mag"],
                        grv=opts["grv"],
                    )
                )

            save_pred(
                lr=lr,
                hr=hr,
                sr=sr,
                gt=gt,
                scale=scale,
                save_path=opts["save_path"],
                prefix=geo_d,
                suffix=suffix,
            )

    return {
        "L1": l1_avg.item(),
        "PSNR": psnr_avg.item(),
    }


def save_pred(
    lr,
    sr,
    hr,
    gt,
    save_path="",
    suffix="",
    scale=None,
    # c_exp: comet_ml.Experiment = None,
    extra="_",
    prefix="pred",
):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    title = f"{prefix}_{suffix}{extra}{scale}x.png"
    norm = Norm(
        clip_min=cfg["test_dataset"]["dataset"]["args"]["norm"][0],
        clip_max=cfg["test_dataset"]["dataset"]["args"]["norm"][1],
    )
    lr = norm.inverse_mmc(lr)
    sr = norm.inverse_mmc(sr)
    hr = norm.inverse_mmc(hr)
    bc = np.array(Image.fromarray(lr).resize(hr.shape, Image.Resampling.BICUBIC))

    # _min, _max = (norm.min,  norm.max)
    _min, _max = (hr.min(), hr.max())
    # _min, _max = (gt.min(), gt.max())

    plt_args = dict(
        vmin=_min,
        vmax=_max,
        cmap=cc.cm.CET_L8,
        # vmin=-20,
        # vmax=40,
        # cmap=cc.cm.CET_R1,
        origin="lower",
        interpolation="nearest",
    )

    fig, [
        [axlr, axbc, axsr, axhr, axgt],
        [axoff1, axdbc, axdsr, axoff2, axoff3],
    ] = plt.subplots(2, 5, figsize=(20, 10))
    plt.suptitle(title)

    axlr.set_title("LR")
    imlr = axlr.imshow(lr, **plt_args)
    plt.colorbar(mappable=imlr, ax=axlr, label="nT", location="bottom")

    axbc.set_title("Bicubic")
    imbc = axbc.imshow(bc, **plt_args)
    plt.colorbar(mappable=imbc, ax=axbc, label="nT", location="bottom")

    axsr.set_title("SR")
    imsr = axsr.imshow(sr, **plt_args)
    plt.colorbar(mappable=imsr, ax=axsr, label="nT", location="bottom")

    axhr.set_title("HR")
    imhr = axhr.imshow(hr, **plt_args)
    plt.colorbar(mappable=imhr, ax=axhr, label="nT", location="bottom")

    axgt.set_title("GT")
    plt_args.pop("vmin")
    plt_args.pop("vmax")
    plt_args["cmap"] = cc.cm.CET_L1
    imgt = axgt.imshow(gt, **plt_args)
    axgt.vlines(
        range(
            0,
            gt.shape[1],
            cfg["test_dataset"]["dataset"]["args"]["hr_line_spacing"] * scale,
        ),
        0,
        gt.shape[0],
        color="r",
        linewidth=1,
    )  # hr_line spacing * scale !!!
    axgt.set_ylim(0, gt.shape[0])
    plt.colorbar(mappable=imgt, ax=axgt, label="nT", location="bottom")

    for ax in [axoff1, axoff2, axoff3]:
        ax.set_axis_off()

    _dmax = max(
        # abs((hr - sr).min()),
        # abs((hr - sr).max()),
        abs((hr - bc).min()),
        abs((hr - bc).max()),
    )

    plt_args = dict(
        cmap=cc.cm.CET_D7,
        origin="lower",
        interpolation="nearest",
        vmin=-_dmax,
        vmax=_dmax,
    )
    axdbc.set_title("HR - BC")
    imdbc = axdbc.imshow(hr - bc, **plt_args)
    plt.colorbar(mappable=imdbc, ax=axdbc, label=r"$\Delta$nT", location="bottom")

    axdsr.set_title("HR - SR")
    imdsr = axdsr.imshow(hr - sr, **plt_args)
    plt.colorbar(mappable=imdsr, ax=axdsr, label=r"$\Delta$nT", location="bottom")

    # lr_ls = ["dataset"]["args"]["hr_line_spacing"] * scale
    plt.savefig(
        Path(save_path) / (title),
        dpi=300,
    )
    plt.close()


def plt_results(results, opts):
    """Plot scatter plot of performance metrics for each scale factor."""
    nlp = np.array(
        [
            (i[0][0], i[1]["L1"], i[1]["PSNR"])
            for i in list([k, v] for k, v in results.items())
        ],
        dtype=float,
    )

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title(f"{opts['set']} set scale-averaged metrics - {opts['geo_d']}")
    ax1.set_xlabel("Scale Factor")
    ax1.plot(nlp[:, 0], nlp[:, 2], "r-")
    ax2.plot(nlp[:, 0], nlp[:, 1], "b--")
    ax1.set_ylabel("PSNR", color="red")
    ax2.set_ylabel("Mean Absolute Error", color="blue")
    # ax2.invert_yaxis()
    plt.savefig(
        Path(opts["save_path"])
        / f"0_Scale_Averaged_Metrics_{opts['geo_d']}_{opts['set']}.png",
        dpi=300,
    )


class CustomTestDataset(HRLRNoddyverse):
    def __init__(self, name, sample, **kwargs):
        self.name = name
        self.sample = sample
        # self.inp_size = input_size,
        self.crop = kwargs.get("crop")
        self.sp = {
            "hr_line_spacing": kwargs.get("hr_line_spacing", 1),
            "sample_spacing": kwargs.get("sample_spacing", 20),
            "heading": kwargs.get("heading", "NS"),  # "EW"
        }
        self.len = len(sample)

    def _process(self, index):
        self.data = {}
        self.data["gt_grid"] = self.sample
        hls = self.sp["hr_line_spacing"]
        lls = int(hls * self.scale)
        hr_x, hr_y, hr_z = self._subsample(self.data["gt_grid"], hls)
        lr_x, lr_y, lr_z = self._subsample(self.data["gt_grid"], lls)
        # # lr dimension: self.inp_size
        # lr_exent = int((self.crop_extent / self.scale) * 4)  # cs_fac = 4
        # lr_e = int(torch.randint(low=0, high=lr_exent - 600, size=(1,)))
        # lr_n = int(torch.randint(low=0, high=lr_exent - 600, size=(1,)))
        # # Note - we use scale here as a factor describing how big HR is x LR.
        # # This diverges from what my brain apparently normally does.
        self.data["hr_grid"] = self._grid(hr_x, hr_y, hr_z, ls=hls, d=self.inp_size)
        self.data["lr_grid"] = self._grid(lr_x, lr_y, lr_z, ls=lls, d=self.inp_size)


def load_real_tifff(p):
    grid = tifffile.imread(p)
    return torch.from_numpy(grid).unsqueeze(0)  # add channel dim


def test_custom_data(model, scale, opts):
    """Run model on custom samples not in existing Dataset
    For now, processes Naprstek synthetic test sample.
    """

    custom = {  # Mapping of test name to normalised C,H,W tensor
        "naprstek": load_naprstek(
            root="D:/luke/Noddy_data/test",
            data_txt_file="Naprstek_BaseModel1-AllValues-1nTNoise.txt",
            normalise=True,
        ),
    }

    for name, sample in custom.items():
        d_args = cfg["test_dataset"]["dataset"]["args"]
        w_args = cfg["test_dataset"]["wrapper"]["args"]

        dset = NoddyverseWrapper(
            CustomTestDataset(
                name,
                sample,
                hr_line_spacing=d_args["hr_line_spacing"],
                sample_spacing=d_args["sample_spacing"],
                heading=d_args["heading"],
            ),
            inp_size=580,
            scale_min=scale,  # also sets scale
            crop=w_args["crop"],
        )

        loader = DataLoader(
            dset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        opts["ids"] = [name]
        opts["shave"] = 25
        opts["gt"] = load_naprstek(
            root="D:/luke/Noddy_data/test",
            data_txt_file="Naprstek_BaseModel1-AllValues-1nTNoise.txt",
            normalise=False,
        )

        return eval(model, scale, loader, opts)


def real_inference(filepath: Path, cfg, scale: float, device="cuda", max_s=128):
    """Run inference on square (only?) grid data without GT.
    Provide cfg as used for training/testing - at least model_dir & model_name.
    max_crop limits grids to RAM-tolerable size
    cpu device unsupported until LTE etc refactored
    """

    filepath = Path(filepath)
    assert filepath.exists(), filepath.absolute()
    nan_val = -99999

    normaliser = Norm(-5000, 5000)
    norm = normaliser.min_max_clip
    unorm = normaliser.inverse_mmc

    model, model_name, _ = load_model(cfg, device)

    grid = norm(load_real_tifff(filepath)).squeeze()  # remove ch dim

    # grid = grid[0:200, 0:200]

    if grid.shape[-2] > max_s or grid.shape[-1] > max_s:
        lr_tiles, tiles_per_row, tiles_per_column = input_tiler(grid, shape=max_s)

    hr_s = max_s * scale
    output_sr = nan_val * np.ones(
        (tiles_per_column * hr_s, tiles_per_row * hr_s), dtype=np.float32
    )
    col_num = 0
    row_num = 0  # top row

    for i, lr in enumerate(tqdm(lr_tiles)):
        lr = lr.unsqueeze(0)  # Add Batch

        b, c, h, w = lr.shape
        h *= scale
        w *= scale

        coord = utils.make_coord((h, w))
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        lr = lr.to(device, non_blocking=True)
        coord = coord.unsqueeze(0).to(device, non_blocking=True)
        cell = cell.unsqueeze(0).to(device, non_blocking=True)

        pred = model(lr, coord, cell).detach().cpu()
        sr = pred.view((b, c, h, w)).permute(0, 2, 3, 1)[:, :, :, 0].contiguous()
        sr = unorm(sr.numpy())

        for j in range(len(lr)):
            rt = hr_s * (row_num + 0)
            rb = hr_s * (row_num + 1)
            cl = hr_s * (col_num + 0)
            cr = hr_s * (col_num + 1)
            output_sr[rt:rb, cl:cr] = sr[j]

            col_num += 1  # move from left to right
            if col_num == tiles_per_row:
                col_num = 0  # move back to left-most column
                row_num += 1  # move down from top to bottom

    output_sr = output_sr[
        : grid.shape[0] * scale,
        : grid.shape[1] * scale,
    ]  # trim padding

    return output_sr


def input_tiler(grid: torch.Tensor, shape=128):
    """Tile grid (Tensor) to shape
    Grid here should be H x W (unbatched)
    Output lr_tiles will be n x C x shape x shape
    Tiles will be padded with 0 (which is the normalised mean) where necessary
    """

    if grid.shape[0] == 1:  # if has a channel dim (n.b. change for 3 ch, etc)
        grid = grid.squeeze()
    lr_s = shape
    grid = torch.nn.functional.pad(
        grid, (0, (lr_s - (grid.shape[-1] % lr_s)), 0, (lr_s - (grid.shape[-2] % lr_s)))
    )

    tiles_per_row = grid.shape[-1] // lr_s
    tiles_per_column = grid.shape[-2] // lr_s

    patches = grid.unfold(0, lr_s, lr_s).unfold(1, lr_s, lr_s)
    lr_tiles = patches.contiguous().view(
        tiles_per_row * tiles_per_column, -1, lr_s, lr_s
    )

    return lr_tiles, tiles_per_row, tiles_per_column


if __name__ == "__main__":
    with open("configs/inference.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    scale = 4
    filepath = "D:/luke/data_source/NSW_80m_test_clip/NSW_80m_test_clip_LR.tif"
    filepath = Path(filepath)

    # results = main()
    sr = real_inference(
        filepath=filepath,
        cfg=cfg,
        scale=scale,
        # device="cuda"
    )

    plt.imshow(sr, origin="lower")
    plt.colorbar()

    with rasterio.open(filepath) as src:
        pred_meta = src.meta
        pred_meta.update(
            {
                "driver": "GTiff",
                "height": sr.shape[0],
                "width": sr.shape[1],
                "transform": src.meta["transform"]
                * src.meta["transform"].scale(1 / scale),
            }
        )

        with rasterio.open(
            f"{filepath.stem}_sr-{scale}x_{cfg['model_name']}.tif", "w", **pred_meta,
        ) as dst:
            dst.write(sr.astype(rasterio.float32), 1)
