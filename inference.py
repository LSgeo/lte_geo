from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import datasets
import models
import utils
from test import save_pred, reshape, batched_predict


def main():
    ### Load Model ###
    model_dir = Path(cfg["model_dir"])
    model_name = cfg["model_name"]
    model_paths = list(model_dir.glob(f"**/*{model_name}*best.pth"))
    if len(model_paths) != 1:
        raise FileNotFoundError(
            f"No unique model found in {model_dir} for *{model_name}*best.pth. Refine search."
        )

    model_spec = torch.load(model_paths[0])["model"]
    model = models.make(model_spec, load_sd=True).cuda()

    ### Define Data ###
    spec = cfg["test_dataset"]
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    # dataset.crop = spec["wrapper"]["args"]["crop"] # this should now be handled in .make()

    if cfg["limit_to_plots"]:
        dataset = Subset(dataset, cfg["plot_samples"])

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        num_workers=cfg.get("num_workers"),
        persistent_workers=bool(cfg.get("num_workers")),
        pin_memory=True,
    )

    ### Pack Options ###
    opts = dict(
        model_name=model_name,
        model_path=model_paths[0],
        save_path=Path(cfg["inference_output_dir"] or f"inference/{model_name}"),
        rgb_range=cfg["rgb_range"],
        shave_factor=3,  # pixels to shave (edges may include NaN)
    )

    scale_min = spec["wrapper"]["args"]["scale_min"]
    scale_max = spec["wrapper"]["args"]["scale_max"]

    print(
        f"\nModel: {opts['model_path'].absolute()}\n"
        f"Saving to: {opts['save_path'].absolute()}"
    )

    ### Do Inference ###
    pbar_m = tqdm(range(scale_min, scale_max + 1))
    for scale in pbar_m:
        pbar_m.set_description(f"Scale: {scale}x")
        dataset.scale = scale
        if cfg["limit_to_plots"]:
            dataset.dataset.scale = scale
        opts["shave"] = scale * opts["shave_factor"]

        results = eval(model, scale, loader, opts)
        pbar_m.write(
            ", ".join(
                f"{metric_name}: {metric_value:.4f}"
                for metric_name, metric_value in results.items()
            )
        )

        # if cfg["custom_grids"]:
        #     test_custom_data(model, scale, cfg, opts)


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
            if opts["cfg"]["eval_bsize"]:
                pred = batched_predict(
                    model, inp, coord, cell, opts["cfg"]["eval_bsize"]
                )
            else:
                pred = model(inp, coord, cell)

        pred, batch = reshape(batch, h_pad=0, w_pad=0, coord=coord, pred=pred)

        l1 = l1_fn(pred, batch["gt"])
        l1_avg.add(l1.item(), inp.shape[0])
        psnr = psnr_fn(pred, batch["gt"])
        psnr_avg.add(psnr.item(), inp.shape[0])

        pbar.set_description(
            f"Test (Mean): L1:{l1_avg.item():.4f}, PSNR:{psnr_avg.item():.4f}"
        )

        if opts["cfg"]["limit_to_plots"]:
            lr = batch["inp"].detach().cpu().squeeze().numpy()
            hr = batch["gt"].detach().cpu().squeeze().numpy()
            sr = pred.detach().cpu().squeeze().numpy()

            save_pred(
                lr=lr,
                hr=hr,
                sr=sr,
                scale=scale,
                gt_index=opts["cfg"]["plot_samples"][i],
                root_path=opts["cfg"]["test_dataset"]["dataset"]["args"]["root_path"],
                save_path=opts["save_path"],
                suffix=opts["cfg"]["plot_samples"][i],
            )

    return {
        "L1": l1_avg.item(),
        "PSNR": psnr_avg.item(),
    }


# def test_custom_data(model, cfg, opts):
#     """Run model on custom samples not in existing Dataset
#     For now, processes Naprstek synthetic test sample.
#     """
#     from datasets.noddyverse import HRLRNoddyverse, NoddyverseWrapper
#     from datasets.noddyverse import load_naprstek_synthetic as load_naprstek

#     class CustomTestDataset(HRLRNoddyverse):
#         def __init__(self, name, sample, **kwargs):
#             self.name = name
#             self.sample = sample
#             self.inp_size = kwargs.get("inp_size")
#             self.crop_extent = kwargs.get("crop_extent")
#             self.crop = bool(self.crop_extent)
#             self.sp = {
#                 "hr_line_spacing": kwargs.get("hr_line_spacing", 1),
#                 "sample_spacing": kwargs.get("sample_spacing", 20),
#                 "heading": kwargs.get("heading", "NS"),  # "EW"
#             }

#         def _process(self, index):
#             self.data = {}
#             self.data["gt_grid"] = self.sample
#             hls = self.sp["hr_line_spacing"]
#             lls = int(hls * self.scale)
#             hr_x, hr_y, hr_z = self._subsample(self.data["gt_grid"], hls)
#             lr_x, lr_y, lr_z = self._subsample(self.data["gt_grid"], lls)
#             # lr dimension: self.inp_size
#             lr_exent = int((self.crop_extent / self.scale) * 4)  # cs_fac = 4
#             lr_e = int(torch.randint(low=0, high=lr_exent - 600, size=(1,)))
#             lr_n = int(torch.randint(low=0, high=lr_exent - 600, size=(1,)))
#             # Note - we use scale here as a factor describing how big HR is x LR.
#             # This diverges from what my brain apparently normally does ().
#             self.data["hr_grid"] = self._grid(
#                 hr_x, hr_y, hr_z, scale=self.scale, ls=hls, lr_e=lr_e, lr_n=lr_n
#             )
#             self.data["lr_grid"] = self._grid(
#                 lr_x, lr_y, lr_z, scale=1, ls=lls, lr_e=lr_e, lr_n=lr_n
#             )

#     synth = {
#         "naprstek": load_naprstek(
#             root="D:/luke/Noddy_data/test",
#             data_txt_file="Naprstek_BaseModel1-AllValues-1nTNoise.txt",
#         ),
#     }
#     dsets = []
#     for name, sample in synth.items():
#         d_args = cfg["test_dataset"]["dataset"]["args"]
#         w_args = cfg["test_dataset"]["wrapper"]["args"]
#         dset = NoddyverseWrapper(
#             CustomTestDataset(
#                 name,
#                 sample,
#                 hr_line_spacing=d_args["hr_line_spacing"],
#                 sample_spacing=d_args["sample_spacing"],
#                 heading=d_args["heading"],
#             ),
#             inp_size=w_args["inp_size"],
#             crop_extent=w_args["crop_extent"],
#         )
#         dset.crop = d_args["test_dataset"]["dataset"]["args"]["crop"]
#         dset.scale = w_args
#         dsets.append(dset)


if __name__ == "__main__":
    with open("configs/inference.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main()
