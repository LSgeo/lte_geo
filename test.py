if 0 and __name__ == "__main__":
    assert False, "Use inference.py instead"

import argparse
import os
import math
from functools import partial
from pathlib import Path

import comet_ml
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import yaml
from PIL import Image
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


def eval_psnr(
    loader,
    model,
    # data_norm=None,
    eval_type=None,
    eval_bsize=None,
    window_size=0,
    scale_max=4,
    fast=False,
    verbose=True,
    rgb_range=1,
    model_name="",
    c_exp: comet_ml.Experiment = None,
    shave=None,
    save_path="",
):
    model.eval()
    scale = int(eval_type.split("-")[1])
    metric_fn = partial(
        utils.calc_psnr,
        dataset="noddyverse",
        scale=scale,
        rgb_range=rgb_range,
        shave=shave,
    )

    loader.dataset.scale = scale
    loader.dataset.dataset.scale = scale

    if __name__ == "__main__":
        loader.dataset.scale_min = scale
        loader.dataset.scale_max = scale

    l1_fn = torch.nn.L1Loss()
    l1_res = utils.Averager()
    val_res = utils.Averager()
    ssim_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc="Eval", dynamic_ncols=True)
    for i, batch in enumerate(pbar):
        loader.dataset.scale = torch.randint(
            low=loader.dataset.scale_min,
            high=loader.dataset.scale_max + 1,
            size=(1,),
        )

        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch["inp"]
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

        # pred.clamp_(0, 1)
        if eval_type is not None and not fast:  # reshape for shaving-eval
            pred, batch = reshape(batch, h_pad, w_pad, coord, pred)
            ssim = piq.ssim(pred.clamp(0, 1), batch["gt"])
            ssim_res.add(ssim.item(), inp.shape[0])

            # if __name__ == "__main__":
            #     save_pred(
            #         lr=batch["inp"].detach().cpu().squeeze().numpy(),
            #         hr=batch["gt"].detach().cpu().squeeze().numpy(),
            #         sr=pred.detach().cpu().squeeze().numpy(),
            #         gt_index=config["plot_samples"][i],
            #         save_path=save_path,
            #         suffix=config["plot_samples"][i],
            #         scale=scale,
            #         c_exp=c_exp,
            #     )

        l1_metric = l1_fn(pred, batch["gt"])
        l1_res.add(l1_metric.item(), inp.shape[0])
        res = metric_fn(pred, batch["gt"])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description("PSNR eval {:.4f}".format(val_res.item()))

    return l1_res.item(), val_res.item(), ssim_res.item()


def reshape(batch, h_pad, w_pad, coord, pred):
    # gt reshape
    b, c, ih, iw = batch["inp"].shape
    s = math.sqrt(batch["coord"].shape[1] / (ih * iw))
    shape = [b, round(ih * s), round(iw * s), c]
    batch["gt"] = batch["gt"].view(*shape).permute(0, 3, 1, 2).contiguous()

    # prediction reshape
    ih += h_pad
    iw += w_pad
    s = math.sqrt(coord.shape[1] / (ih * iw))
    shape = [b, round(ih * s), round(iw * s), c]
    pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
    pred = pred[..., : batch["gt"].shape[-2], : batch["gt"].shape[-1]]

    return pred, batch


def plot_scale_range(loader, model, scales=[1, 2, 3, 4], model_name=""):
    raise NotImplementedError
    model.eval()
    pbar = tqdm(scales, leave=False, desc="scale_vis")

    for scale in pbar:
        loader.dataset.scale = scale
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

            # save_pred(
            #     lr=batch["inp"],
            #     sr=pred,
            #     hr=batch["gt"],
            #     gt_index=config["visually_nice_test_samples"][i],
            #     save_path=f"inference/scale_vis/{model_name}",
            #     suffix=config["visually_nice_test_samples"][i],
            #     scale=scale,
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_swinir-lte_geo.yaml")
    parser.add_argument(
        "--model", default="D:/luke/lte_geo/save/_train_swinir-lte_geo"
    )  # Specify dir containing model .pths
    parser.add_argument("--window", default="0")
    parser.add_argument("--scale_max", default="4")
    parser.add_argument("--fast", default=False)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # process_custom_data()
    # exit()

    spec = config["test_dataset"]
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    # print(len(dataset))
    dataset.crop = spec["wrapper"]["args"]["crop"]

    if config["limit_to_plots"]:
        dataset = Subset(dataset, config["plot_samples"])

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        num_workers=config.get("num_workers"),
        persistent_workers=bool(config.get("num_workers")),
        pin_memory=True,
    )

    model_dir = Path(args.model)
    model_name = config["model_name"]
    model_paths = list(model_dir.glob(f"**/*{model_name}*best.pth"))
    if len(model_paths) != 1:
        raise FileNotFoundError(
            f"No unique model found in {model_dir} for *{model_name}*best.pth. Refine search."
        )
    model_path = model_paths[0]
    # last_model = Path("D:/luke/lte_geo/save/_train_swinir-lte_geo/tensorboard")
    # last_model = sorted(list(last_model.iterdir()))[-1].stem

    model_spec = torch.load(model_path)["model"]
    model = models.make(model_spec, load_sd=True).cuda()

    save_path = Path(f"inference/{model_name}")

    if config["scale_range"]:
        plot_scale_range(loader, model, scales=[1, 2, 3, 4], model_name=model_name)
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
            model_name=model_name,
            shave=6,
            save_path=save_path,
        )
        print(
            f"Model: {model_path.absolute()}\n"
            f"L1: {res[0]:.4f} PSNR: {res[1]:.4f}\n"
            f"Saved to: {save_path}"
        )
