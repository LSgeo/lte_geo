# modified from: https://github.com/yinboc/liif

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
import yaml

from comet_ml import Experiment
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ch2.ltegeo import datasets
from ch2.ltegeo import models
from ch2.ltegeo import utils
from ch2.ltegeo.test import reshape, eval_psnr

plt.ioff()


def make_data_loader(spec, tag=""):
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    log(f"{tag} dataset:")

    if "preview" in tag:
        dataset = Subset(dataset, config["plot_samples"])
        dataset.dataset.preview_mode = True
        dataset.dataset.sample_q = None  # Preview full extent
        bs = 1
        num_workers = 0
        log(f"  Preview Scales: {dataset.dataset.scales}")
    else:
        bs = spec["batch_size"]
        num_workers = config.get("num_workers")
        log(f"  Scales: {dataset.scales}")
    log(f"  Size: {len(dataset)}")

    # for k, v in dataset[0].items():
    #     log("  {}: shape={}".format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=(tag == "train"),
        num_workers=num_workers,
        persistent_workers=bool(num_workers),
        pin_memory=True,
    )

    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get("train_dataset"), tag="train")
    val_loader = make_data_loader(config.get("val_dataset"), tag="val")
    preview_loader = make_data_loader(config.get("val_dataset"), tag="preview")

    return train_loader, val_loader, preview_loader


def prepare_training(train_loader, val_loader, preview_loader):
    if config.get("resume") is not None:
        # Find and load saved model
        best_or_last = "last"
        model_dir = Path(config["model_dir"])
        model_name = config["resume"]
        model_paths = list(model_dir.glob(f"**/*{model_name}*{best_or_last}.pth"))
        if len(model_paths) != 1:
            raise FileNotFoundError(
                f"No unique model found in {model_dir.absolute()}"
                f"for *{model_name}*{best_or_last}.pth."
            )
        resume = model_paths[0]
        sv_file = torch.load(resume)
        model = models.make(sv_file["model"], load_sd=True).to(
            "cuda", non_blocking=True
        )
        if config.get("only_resume_weights"):  # Start "new" training
            optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
            epoch_start = 1
            lr_scheduler = MultiStepLR(
                optimizer, **config.get("scheduler")["multi_step_lr"]
            )
        else:  # Resume training
            optimizer = utils.make_optimizer(
                model.parameters(), sv_file["optimizer"], load_sd=True
            )
            epoch_start = sv_file["epoch"] + 1
            if "multi_step_lr" in config.get("scheduler"):
                lr_scheduler = MultiStepLR(
                    optimizer, **config.get("scheduler")["multi_step_lr"]
                )

        val_l1, val_res, ssim_res = eval_psnr(
            val_loader,
            model,
            eval_type=config.get("eval_type"),
            eval_bsize=config.get("eval_bsize"),
            c_exp=c_exp,
            shave=config.get("shave"),
        )
        c_exp.log_metric("L1 loss Val", val_l1, step=0)
        c_exp.log_metric("PSNR Val", val_res, step=0)
        c_exp.log_metric("SSIM Val", ssim_res, step=0)
        log_images(preview_loader, model, c_exp)

    else:
        # New model, new training
        model = models.make(config["model"]).to("cuda", non_blocking=True)
        optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
        epoch_start = 1
        if "multi_step_lr" in config.get("scheduler"):
            lr_scheduler = MultiStepLR(
                optimizer, **config.get("scheduler")["multi_step_lr"]
            )
        elif "one_cycle_lr" in config.get("scheduler"):
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=float(config.get("scheduler")["one_cycle_lr"]["max_lr"]),
                total_steps=len(train_loader),
            )
        else:
            raise NotImplementedError("Misconfigured Scheduler")

    log(f"sched: {yaml.dump(config.get('scheduler'))}")
    log("model: #params={}".format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


@torch.no_grad()
def plt_comet(ims: list, names: list, ax_args: dict, c_exp: Experiment):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        im = ims[i].detach().cpu().squeeze().numpy()
        ax.set_title(names[i])
        cax = ax.imshow(im, cmap=cc.cm.CET_L1, interpolation="nearest", **ax_args)
        ax.set_yticks(np.linspace(0, im.shape[0], 6))
        ax.set_xticks(np.linspace(0, im.shape[1], 6))
        plt.colorbar(cax, ax=ax, orientation="horizontal")

    c_exp.log_figure(figure=plt, figure_name=names[0][:-3] + names[-1][-4:])
    plt.close()


def log_images(loader, model, c_exp: Experiment):
    model.eval()
    scales = [8]
    scales.extend(config["val_dataset"]["wrapper"]["args"]["scales"])

    for si in tqdm(range(len(loader.dataset))):
        # NOTE we are not using the dataloader!

        for scale in set(scales):
            n = f"sample_{config['plot_samples'][si]:03d}"
            loader.dataset.dataset.override_scale = scale
            sample = loader.dataset[si]

            inp = sample["inp"].to("cuda", non_blocking=True).unsqueeze(0)
            coord = sample["coord"].to("cuda", non_blocking=True).unsqueeze(0)
            cell = sample["cell"].to("cuda", non_blocking=True).unsqueeze(0)

            pred = model(inp, coord, cell)
            sample = {"inp": inp, "coord": coord, "cell": cell, "gt": sample["gt"]}
            pred, sample = reshape(sample, 0, 0, coord, pred)

            ax_args = (
                {}
            )  # dict(vmin=sample["gt"].min().item(), vmax=sample["gt"].max().item())
            ims = [inp, pred, sample["gt"]]
            names = [
                n + "_lr",
                n + f"_sr_{int(scale):02d}x",
                n + f"_hr_{int(scale):02d}x",
            ]

            plt_comet(ims, names, ax_args, c_exp)


def train_with_fake_epochs(
    train_loader,
    model,
    optimizer,
    scaler,
    val_loader,
    preview_loader,
    epoch_start,
    lr_scheduler,
    save_path,
):
    """If we use the 1M noddy set, we don't use epochs. Instead,
    we iterate the dataset and trigger "epoch behaviour" every n steps.
    i.e. run validation, save model, adjust LR schedule etc.
    """

    ngpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    max_val_v = -1e18
    iter_per_epoch = int(len(train_loader) // config["epoch_max"])
    timer = utils.Timer()

    epoch_max = config["epoch_max"]
    epoch_val = config.get("epoch_val")
    epoch_save = config.get("epoch_save")

    def fake_epoch_start(new_epoch):
        """"""
        t_epoch_start = timer.t()
        log_info = [f"epoch {new_epoch}/{epoch_max}"]
        c_exp.set_epoch(new_epoch)
        return new_epoch, t_epoch_start, log_info

    def fake_epoch_end(curr_epoch, train_loss_itm, max_val_v):
        """"""
        log_info.append(f"train: loss={train_loss_itm:.4f}")
        # writer.add_scalars('loss', {'train': train_loss_itm}, epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], curr_epoch)
        if "multi_step_lr" in config.get("scheduler") or "one_cycle_lr" in config.get(
            "scheduler"
        ):
            c_exp.log_metric("LR", lr_scheduler.get_last_lr()[0], epoch=curr_epoch)
            # lr_scheduler.step()
        else:
            raise NotImplementedError("Misconfigured Scheduler")

        if ngpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config["model"]
        model_spec["sd"] = model_.state_dict()
        optimizer_spec = config["optimizer"]
        optimizer_spec["sd"] = optimizer.state_dict()
        scaler_spec = {"sd": scaler.state_dict()}
        sv_file = {
            "model": model_spec,
            "optimizer": optimizer_spec,
            "epoch": curr_epoch,
            "scaler": scaler_spec,
        }

        # Save every epoch as "-last"
        torch.save(sv_file, save_path / f"{c_exp.get_name()}_epoch-last.pth")

        # Save every n epoch as "epoch-n" checkpoint
        if (epoch_save is not None) and (curr_epoch % epoch_save == 0):
            torch.save(
                sv_file, save_path / f"{c_exp.get_name()}_epoch-{curr_epoch}.pth"
            )

        # Run Validation, and save best model
        if (epoch_val is not None) and (curr_epoch % epoch_val == 0):
            if ngpus > 1 and (config.get("eval_bsize") is not None):
                model_ = model.module
            else:
                model_ = model

            val_l1, val_res, extra_res = eval_psnr(
                val_loader,
                model_,
                eval_type=config.get("eval_type"),
                eval_bsize=config.get("eval_bsize"),
                c_exp=c_exp,
                shave=config.get("shave"),
            )

            log_images(preview_loader, model_, c_exp)

            log_info.append(f"val: psnr={val_res:.4f}")
            c_exp.log_metric("L1 loss Val", val_l1, epoch=curr_epoch)
            c_exp.log_metric("PSNR Val", val_res, epoch=curr_epoch)

            for res, val in extra_res.items():
                c_exp.log_metric(f"{res.upper()} Val", val, epoch=curr_epoch)

            # c_exp.log_metric("SSIM Val", ssim_res)
            # writer.add_scalars('psnr', {'val': val_res}, curr_epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, save_path / f"{c_exp.get_name()}_epoch-best.pth")

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append(f"{t_epoch} {t_elapsed}/{t_all}")

        epoch_pbar.write(", ".join(log_info))  # log.write(...)
        writer.flush()

        return max_val_v

    # iteration = (epoch - 1) * iter_per_epoch + i
    epoch = epoch_start
    epoch_pbar = tqdm(
        total=config.get("epoch_max"), desc="Epoch", leave=True, unit="epoch"
    )

    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr
    l1_loss_fn = nn.L1Loss()
    # if True:  # "fsim" in config["loss_fn"]:
    #     fsim_loss_fn = (
    #         piq.fsim
    #     )  # FSIM_Loss(data_range=config["rgb_range"], chromatic=False)
    # if True:  # "ssim" in config["loss_fn"]:
    #     ssim_loss_fn = piq.ssim  # SSIM_Loss(data_range=config["rgb_range"])
    # elif "haarpsi" in config["loss_fn"]:
    #     iqa_loss_fn = piq.HaarPSILoss(
    #         data_range=config["rgb_range"],
    #         c=20,  # selected for higher "noise" task performance
    #         alpha=6,  # selected for higher "noise" task performance
    #     )
    # elif "l1" in config["loss_fn"]:
    #     pass
    # else:
    #     raise NotImplementedError(f'{config["loss_fn"]} unsupported')

    for iteration, batch in enumerate(
        tqdm(train_loader, leave=True, desc="Train iteration", dynamic_ncols=True),
        start=1,  # to match iter_per_epoch, etc
    ):
        if (iteration == 1) or (iteration % (iter_per_epoch + 1) == 0):
            epoch, t_epoch_start, log_info = fake_epoch_start(epoch)
        c_exp.set_step(iteration)

        model.train()
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        with autocast(enabled=config.get("use_amp_autocast", False)):
            # OK so. coord and gt are the coord/val lists for HR.
            # Cell is something to do with Fourier/Phase info
            pred = model(batch["inp"], batch["coord"], batch["cell"])
            l1_loss = l1_loss_fn(pred, batch["gt"])

            # Reshape is useless when using random sample_q
            # we don't know what samples are where!

            if False:
                pred, batch = reshape(batch, 0, 0, batch["coord"], pred)
                shave = config.get("shave")
                shave = int((shave / 100) * batch["gt"].shape[-1])
                pred = pred[..., shave:-shave, shave:-shave]
                gt = batch["gt"][..., shave:-shave, shave:-shave]
                pred.clamp_(0.0, 1.0)
                gt.clamp_(0.0, 1.0)
                psnr = metric_fn(
                    pred,
                    gt,
                    rgb_range=config.get("rgb_range"),
                    shave=config.get("shave"),
                )
                # if pred.min() < -0 or pred.max() > 1:
                #     log(f"Value of {pred.min()=} {pred.max()=} out of range")
                fsim_loss = fsim_loss_fn(pred, gt, chromatic=False).item()
                ssim_loss = ssim_loss_fn(pred, gt).item()
                c_exp.log_metric("FSIM Train", fsim_loss, step=iteration)
                c_exp.log_metric("SSIM Train", ssim_loss, step=iteration)
            # iqa_loss = iqa_loss_fn(pred, gt)
            # loss = iqa_loss
            # iqa_loss_item = iqa_loss.item()
            # else:
            #     raise ValueError(f"Loss function not supported, {epoch=}")

        scaler.scale(l1_loss).backward()
        scaler.step(optimizer)  # .step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()  # For OneCycleLR

        l1_loss_item = l1_loss.item()
        # psnr_item = psnr.item()
        train_loss.add(l1_loss_item)  # loss.item())

        writer.add_scalars("L1 loss Train", {"train": l1_loss_item}, iteration)
        # writer.add_scalars("PSNR", {"train": psnr_item}, iteration)
        c_exp.log_metric("L1 loss Train", l1_loss_item, step=iteration)
        # c_exp.log_metric("PSNR Train", psnr_item, step=iteration)
        # if (
        #     "fsim" in config["loss_fn"]
        #     or "haarpsi" in config["loss_fn"]
        #     or "ssim" in config["loss_fn"]
        # ):
        #     writer.add_scalars(
        #         f"{config['loss_fn']} loss Train", {"train": iqa_loss_item}, iteration
        #     )
        #     c_exp.log_metric(
        #         f"{config['loss_fn']} loss Train", iqa_loss_item, step=iteration
        #     )

        if iteration % iter_per_epoch == 0:
            max_val_v = fake_epoch_end(epoch, train_loss.item(), max_val_v)
            epoch_pbar.update()
            epoch += 1

    epoch_pbar.update()
    epoch_pbar.close()

    return None


def main(config_, save_path):
    global config, log, writer, c_exp
    config = config_
    c_exp = Experiment(
        disabled=not config["use_comet"],
        auto_output_logging="native",
    )
    torch.manual_seed(21)
    np.random.seed(21)
    random.seed(21)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    save_path = (
        Path(save_path) / f'{datetime.now().strftime("%y%m%d-%H%M")}_{c_exp.get_name()}'
    )
    save_path.mkdir(exist_ok=False, parents=True)
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(save_path / f"{c_exp.get_name()}_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader, preview_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training(
        train_loader, val_loader, preview_loader
    )
    scaler = GradScaler(enabled=config.get("use_amp_scaler", False))

    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    tags = []
    tags.extend([config.get("model")["name"]])
    tags.extend(["amp_scaler"] if config.get("use_amp_scaler") else [])
    tags.extend(["amp_autocast"] if config.get("use_amp_autocast") else [])
    tags.extend(["resume", config.get("resume")] if config.get("resume") else [])
    tags.extend(
        ["real_data"]
        if "real_dataset" in config["train_dataset"]["dataset"]["name"]
        else []
    )
    tags.extend(
        ["aug"]
        if any(config["train_dataset"]["wrapper"]["args"].get("augmentations").values())
        else []
    )  # TODO FIX - returns true even if all augs are false
    scale_tags = [f"{s}x" for s in config["train_dataset"]["wrapper"]["args"]["scales"]]
    tags.extend(scale_tags)
    tags.extend([config["loss_fn"].upper()])
    if config["train_dataset"]["dataset"]["args"].get("events") is not None:
        event_tags = [
            f"{e}" for e in config["train_dataset"]["dataset"]["args"]["events"]
        ]
        tags.extend(event_tags)
    c_exp.add_tags(tags)

    c_exp.log_parameters(flatten_dict(config))
    c_exp.log_code("ch2/ltegeo/datasets/noddyverse.py")

    train_with_fake_epochs(
        train_loader,
        model,
        optimizer,
        scaler,
        val_loader,
        preview_loader,
        epoch_start,
        lr_scheduler,
        save_path,
    )

    c_exp.end()


def flatten_dict(cfg, sep=" | "):
    """Flatten a 4-nested dictionary with {sep} as a separator for keys"""
    flat_dict = {}
    for k0, v0 in cfg.items():
        try:
            for k1, v1 in v0.items():
                try:
                    for k2, v2 in v1.items():
                        try:
                            for k3, v3 in v2.items():
                                try:
                                    for k4, v4 in v3.items():
                                        flat_dict[
                                            f"{k0}{sep}{k1}{sep}{k2}{sep}{k3}{sep}{k4}"
                                        ] = v4
                                except AttributeError:
                                    flat_dict[f"{k0}{sep}{k1}{sep}{k2}{sep}{k3}"] = v3
                        except AttributeError:
                            flat_dict[f"{k0}{sep}{k1}{sep}{k2}"] = v2
                except AttributeError:
                    flat_dict[f"{k0}{sep}{k1}"] = v1
        except AttributeError:
            flat_dict[f"{k0}"] = v0

    return flat_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="ch2/ltegeo/configs/train_swinir-lte_geo_synthetic.yaml"
        # default="ch2/ltegeo/configs/train_swinir-lte_geo_real.yaml",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Config loaded from {args.config}")

    save_name = args.name
    if save_name is None:
        save_name = "_" + args.config.split("/")[-1][: -len(".yaml")]
    if args.tag is not None:
        save_name += "_" + args.tag
    save_path = os.path.join("ch2/ltegeo/save", save_name)

    main(config, save_path)
    print("Finished.")
