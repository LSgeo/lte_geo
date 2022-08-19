# modified from: https://github.com/yinboc/liif

import argparse
import os

from comet_ml import Experiment
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils


def make_data_loader(spec, tag=""):
    if spec is None:
        return None

    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})

    log("{} dataset: size={}".format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log("  {}: shape={}".format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=(tag == "train"),
        num_workers=config.get("num_workers"),
        persistent_workers=bool(config.get("num_workers")),
        pin_memory=True,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get("train_dataset"), tag="train")
    val_loader = make_data_loader(config.get("val_dataset"), tag="val")
    return train_loader, val_loader


def prepare_training():
    if config.get("resume") is not None:
        # if os.path.exists(config.get("resume")):
        sv_file = torch.load(config["resume"])
        model = models.make(sv_file["model"], load_sd=True).to(
            "cuda", non_blocking=True
        )
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file["optimizer"], load_sd=True
        )
        epoch_start = sv_file["epoch"] + 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config["model"]).to("cuda", non_blocking=True)
        optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
        epoch_start = 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])

    log("model: #params={}".format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    data_norm = config["data_norm"]
    t = data_norm["inp"]
    inp_sub = (
        torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).to("cuda", non_blocking=True)
    )
    inp_div = (
        torch.FloatTensor(t["div"]).view(1, -1, 1, 1).to("cuda", non_blocking=True)
    )
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).to("cuda", non_blocking=True)
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).to("cuda", non_blocking=True)

    num_dataset = config.get("train_dataset")["dataset"]["args"]["limit_length"]
    iter_per_epoch = int(
        num_dataset
        / config.get("train_dataset")["batch_size"]
        * config.get("train_dataset")["dataset"]["args"]["repeat"]
    )
    iteration = 0
    for batch in tqdm(train_loader, leave=False, desc="train"):
        c_exp.set_step(iteration + iter_per_epoch * epoch)
        # Set scale for next batch
        train_loader.dataset.scale = torch.randint(
            low=train_loader.dataset.scale_min,
            high=train_loader.dataset.scale_max + 1,
            size=(1,),
        )
        while train_loader.dataset.scale == 7:  # or (self.scale == 8: # if cs_fac=5)
            train_loader.dataset.scale = torch.randint(
                low=train_loader.dataset.scale_min,
                high=train_loader.dataset.scale_max + 1,
                size=(1,),
            )
            print(f"set scale to {train_loader.dataset.scale} instead of 7")

        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        inp = (batch["inp"] - inp_sub) / inp_div
        pred = model(inp, batch["coord"], batch["cell"])

        gt = (batch["gt"] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        psnr = metric_fn(pred, gt, rgb_range=config.get("rgb_range"))

        # tensorboard
        writer.add_scalars(
            "loss", {"train": loss.item()}, (epoch - 1) * iter_per_epoch + iteration
        )
        writer.add_scalars(
            "psnr", {"train": psnr}, (epoch - 1) * iter_per_epoch + iteration
        )
        c_exp.log_metric("L1 loss", loss.item())
        c_exp.log_metric("PSNR", psnr)
        iteration += 1

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None
        loss = None

    return train_loss.item()


def main(config_, save_path):
    from test import eval_psnr

    global config, log, writer, c_exp
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)
    c_exp = Experiment(disabled=not config["use_comet"])

    train_loader, val_loader = make_data_loaders()
    if config.get("data_norm") is None:
        config["data_norm"] = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config["epoch_max"]
    epoch_val = config.get("epoch_val")
    epoch_save = config.get("epoch_save")
    max_val_v = -1e18

    c_exp.add_tags([])
    c_exp.log_parameters(flatten_dict(config))

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ["epoch {}/{}".format(epoch, epoch_max)]
        c_exp.set_epoch(epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        c_exp.log_metric("lr", optimizer.param_groups[0]["lr"])

        train_loss = train(train_loader, model, optimizer, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append("train: loss={:.4f}".format(train_loss))
        # writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config["model"]
        model_spec["sd"] = model_.state_dict()
        optimizer_spec = config["optimizer"]
        optimizer_spec["sd"] = optimizer.state_dict()
        sv_file = {"model": model_spec, "optimizer": optimizer_spec, "epoch": epoch}

        torch.save(sv_file, os.path.join(save_path, "epoch-last.pth"))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, "epoch-{}.pth".format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get("eval_bsize") is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(
                val_loader,
                model_,
                data_norm=config["data_norm"],
                eval_type=config.get("eval_type"),
                eval_bsize=config.get("eval_bsize"),
                exp=c_exp,
            )

            log_info.append("val: psnr={:.4f}".format(val_res))
            c_exp.log_metric("val_psnr", val_res)
            # writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(
                    sv_file,
                    os.path.join(save_path, f"{c_exp.get_name()}_epoch-best.pth"),
                )

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

        log(", ".join(log_info))
        writer.flush()


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
                                except:
                                    flat_dict[f"{k0}{sep}{k1}{sep}{k2}{sep}{k3}"] = v3
                        except:
                            flat_dict[f"{k0}{sep}{k1}{sep}{k2}"] = v2
                except:
                    flat_dict[f"{k0}{sep}{k1}"] = v1
        except:
            flat_dict[f"{k0}"] = v0

    return flat_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="D:/luke/lte_geo/configs/train_swinir-lte_geo.yaml"
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("config loaded.")

    save_name = args.name
    if save_name is None:
        save_name = "_" + args.config.split("/")[-1][: -len(".yaml")]
    if args.tag is not None:
        save_name += "_" + args.tag
    save_path = os.path.join("./save", save_name)

    main(config, save_path)
