import argparse
import os
import math
from functools import partial
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets
import models
import utils
from datasets.noddyverse import HRLRNoddyverse
from mlnoddy.datasets import Norm, parse_geophysics


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def save_pred(lr, sr, hr, gt_index, save_path="", suffix="", scale=None):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    title = f"Magnetics_{suffix}_{scale}x.png"
    norm = Norm(clip=5000)
    lr = norm.inverse_mmc(lr.detach().cpu().squeeze().numpy())
    sr = norm.inverse_mmc(sr.detach().cpu().squeeze().numpy())
    hr = norm.inverse_mmc(hr.detach().cpu().squeeze().numpy())

    gt_list = Path(spec["dataset"]["args"]["root_path"])
    gt_list = [Path(str(p)[:-3]) for p in gt_list.glob("**/*.mag.gz")]
    gt = next(parse_geophysics(gt_list[gt_index], mag=True))

    # _min, _max = (norm.min,  norm.max)
    _min, _max = (hr.min(), hr.max())
    # _min, _max = (gt.min(), gt.max())

    plt_args = dict(
        vmin=_min,
        vmax=_max,
        cmap=cc.cm.CET_L8,
        origin="lower",
        interpolation="nearest",
    )

    fig, [axlr, axsr, axhr, axgt] = plt.subplots(1, 4, figsize=(24, 8))
    plt.suptitle(title)
    axlr.set_title("LR")
    imlr = axlr.imshow(lr, **plt_args)
    plt.colorbar(mappable=imlr, ax=axlr, location="bottom")
    axsr.set_title("SR")
    imsr = axsr.imshow(sr, **plt_args)
    plt.colorbar(mappable=imsr, ax=axsr, location="bottom")
    axhr.set_title("HR")
    imhr = axhr.imshow(hr, **plt_args)
    plt.colorbar(mappable=imhr, ax=axhr, location="bottom")
    axgt.set_title("GT")
    plt_args.pop("vmin")
    plt_args.pop("vmax")
    plt_args["cmap"] = cc.cm.CET_L1
    imgt = axgt.imshow(gt, **plt_args)
    plt.colorbar(mappable=imgt, ax=axgt, location="bottom")
    # lr_ls = ["dataset"]["args"]["hr_line_spacing"] * scale
    plt.savefig(Path(save_path) / title)
    plt.close()


def eval_psnr(
    loader,
    model,
    data_norm=None,
    eval_type=None,
    eval_bsize=None,
    window_size=0,
    scale_max=4,
    fast=False,
    verbose=False,
    rgb_range=1,
    model_name="",
):
    model.eval()

    if data_norm is None:
        data_norm = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith("div2k"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(utils.calc_psnr, dataset="div2k", scale=scale)
    elif eval_type.startswith("benchmark"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(utils.calc_psnr, dataset="benchmark", scale=scale)
    elif eval_type.startswith("noddy"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(
            utils.calc_psnr,
            dataset="noddyverse",
            scale=scale,
            rgb_range=rgb_range,
        )
        loader.dataset.dataset.scale = scale  # Dataloader>Subset>Wrapper>NoddyDataset
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc="test")
    for i, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()

        # pbar.write(f"Running scale: {loader.dataset.dataset.scale}")

        inp = (batch["inp"] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, : h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, : w_old + w_pad]

            coord = (
                utils.make_coord((scale * (h_old + h_pad), scale * (w_old + w_pad)))
                .unsqueeze(0)
                .cuda()
            )
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0

            coord = batch["coord"]
            cell = batch["cell"]

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            if fast:
                pred = model(inp, coord, cell * max(scale / scale_max, 1))
            else:
                pred = batched_predict(
                    model, inp, coord, cell * max(scale / scale_max, 1), eval_bsize
                )  # cell clip for extrapolation

        pred = pred * gt_div + gt_sub
        # pred.clamp_(0, 1)

        if eval_type is not None and fast == False:  # reshape for shaving-eval
            pred, batch = reshape(batch, h_pad, w_pad, coord, pred)

            save_pred(
                lr=batch["inp"],
                sr=pred,
                hr=batch["gt"],
                gt_index=config["visually_nice_test_samples"][i],
                save_path=f"inference/{model_name}",
                suffix=config["visually_nice_test_samples"][i],
                scale=scale,
            )

        res = metric_fn(pred, batch["gt"])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description("PSNR test {:.4f}".format(val_res.item()))

    return val_res.item()


def reshape(batch, h_pad, w_pad, coord, pred):
            # gt reshape
            ih, iw = batch["inp"].shape[-2:]
            s = math.sqrt(batch["coord"].shape[1] / (ih * iw))
            shape = [batch["inp"].shape[0], round(ih * s), round(iw * s), 1]
            batch["gt"] = batch["gt"].view(*shape).permute(0, 3, 1, 2).contiguous()

            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch["inp"].shape[0], round(ih * s), round(iw * s), 1]
            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            pred = pred[..., : batch["gt"].shape[-2], : batch["gt"].shape[-1]]

    return pred, batch


def single_sample_scale_range(loader, model, scales=[1, 2, 3, 4], model_name=""):
    model.eval()
    pbar = tqdm(scales, leave=False, desc="scale_vis")

    for scale in pbar:
        loader.dataset.dataset.scale = scale

        for i, batch in enumerate(loader):
            for k, v in batch.items():
                batch[k] = v.cuda()
            h_pad = 0
            w_pad = 0

            inp = batch["inp"]
            coord = batch["coord"]
            cell = batch["cell"]

            with torch.no_grad():
                pred = model(inp, coord, cell)

            pred, batch = reshape(batch, h_pad, w_pad, coord, pred)

            save_pred(
                lr=batch["inp"],
                sr=pred,
                hr=batch["gt"],
                gt_index=config["visually_nice_test_samples"][i],
                save_path=f"inference/scale_vis/{model_name}",
                suffix=config["visually_nice_test_samples"][i],
                scale=scale,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_swinir-lte_geo.yaml")
    parser.add_argument("--model", default="save/_train_swinir-lte_geo/epoch-best.pth")
    parser.add_argument("--window", default="0")
    # parser.add_argument("--scale_max", default="4")
    parser.add_argument("--fast", default=False)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args.scale_max = config["test_dataset"]["wrapper"]["args"]["scale_max"]

    spec = config["test_dataset"]
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    dataset.is_val = True

    if True:  # config["show_scale_samples_not_eval"]:
        dataset = Subset(dataset, config["visually_nice_test_samples"])

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        num_workers=config.get("num_workers"),
        persistent_workers=bool(config.get("num_workers")),
        pin_memory=True,
    )

    model_spec = torch.load(args.model)["model"]
    model = models.make(model_spec, load_sd=True).cuda()

    last_model = Path("D:/luke/lte_geo/save/_train_swinir-lte_geo/tensorboard")
    last_model = sorted(list(last_model.iterdir()))[-1].stem

    if config["show_scale_samples_not_eval"]:
        single_sample_scale_range(
            loader, model, scales=[1, 2, 3, 4], model_name=last_model
        )
    else:
    res = eval_psnr(
        loader,
        model,
        data_norm=config.get("data_norm"),
        eval_type=config.get("eval_type"),
        eval_bsize=config.get("eval_bsize"),
        window_size=int(args.window),
        scale_max=int(args.scale_max),
        fast=args.fast,
        verbose=True,
        rgb_range=config.get("rgb_range"),
        model_name=last_model,
    )
    print("result: {:.4f}".format(res))
