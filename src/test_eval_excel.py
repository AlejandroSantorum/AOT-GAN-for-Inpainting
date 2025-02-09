import importlib
import os
import re

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

from utils.option import args
from data.ixi_dataset import IXI_TEST_SUBJECTS_1
from utils.metrics import mse_2d, snr_2d, psnr_2d, ssim_2d


def main(args, use_gpu=True):
    brain_input_dir = os.path.join(args.dir_image, args.data_train)
    mask_input_dir = os.path.join(args.dir_mask, args.mask_type)

    # Model and version
    net = importlib.import_module("model." + args.model)
    if use_gpu:
        model = net.InpaintGenerator(args).cuda()
        model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
    else:
        print(f"Loading model '{args.pre_train}' ...")
        model = net.InpaintGenerator(args)
        model.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
    model.eval()

    subject_names, slice_indices = [], []
    mse_list, snr_list, psnr_list, ssim_list = [], [], [], []
    for subject_name in tqdm(os.listdir(brain_input_dir)):
        if os.path.isdir(os.path.join(brain_input_dir, subject_name)) and subject_name in IXI_TEST_SUBJECTS_1:
            for slice_img_fname in os.listdir(os.path.join(brain_input_dir, subject_name)):
                if slice_img_fname.endswith(".png"):
                    slice_number = int(re.search(r'slice(\d+)', slice_img_fname).group(1))
                    brain_slice_fname = slice_img_fname
                    mask_slice_fname = slice_img_fname.replace("-T1-brain-", "-T1-mask-")
                    # Brain and mask slices filepaths
                    brain_slice_fpath = os.path.join(brain_input_dir, subject_name, brain_slice_fname)
                    mask_slice_fpath = os.path.join(mask_input_dir, subject_name, mask_slice_fname)

                    # Load brain and mask slices
                    brain_slice_img = ToTensor()(Image.open(brain_slice_fpath).convert("RGB"))
                    mask_slice_img = ToTensor()(Image.open(mask_slice_fpath).convert("L"))

                    # Add batch dimension
                    brain_slice_img = (brain_slice_img * 2.0 - 1.0).unsqueeze(0)
                    mask_slice_img = mask_slice_img.unsqueeze(0)

                    # Masking brain slice image with mask slice image
                    masked_brain_slice_img = brain_slice_img * (1 - mask_slice_img.float()) + mask_slice_img

                    if use_gpu:
                        brain_slice_img = brain_slice_img.cuda()
                        mask_slice_img = mask_slice_img.cuda()
                        masked_brain_slice_img = masked_brain_slice_img.cuda()

                    # Inpainting
                    with torch.no_grad():
                        pred_img = model(masked_brain_slice_img, mask_slice_img)

                    # Getting numpy arrays: 0 for batch dimension + cpu() + numpy() + 0 for channel dimension
                    groundtruth_slice_npy = brain_slice_img[0].cpu().numpy()[0]
                    mask_slice_npy = mask_slice_img[0].cpu().numpy()[0]
                    pred_img_npy = pred_img[0].cpu().numpy()[0]

                    # Performance metrics
                    mse = mse_2d(test_img=pred_img_npy, ref_img=groundtruth_slice_npy, mask=mask_slice_npy)
                    snr = snr_2d(test_img=pred_img_npy, ref_img=groundtruth_slice_npy, mask=mask_slice_npy)
                    psnr = psnr_2d(test_img=pred_img_npy, ref_img=groundtruth_slice_npy, mask=mask_slice_npy)
                    ssim = ssim_2d(test_img=pred_img_npy, ref_img=groundtruth_slice_npy, mask=mask_slice_npy)

                    subject_names.append(subject_name)
                    slice_indices.append(slice_number)
                    mse_list.append(mse)
                    snr_list.append(snr)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
    
    mse_list = np.array(mse_list)
    snr_list = np.array(snr_list)
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)

    print(f"Dropping {np.sum(np.isnan(mse_list))} NaN values from MSE array")
    pr_mse_list = mse_list[~np.isnan(mse_list)]
    print(f"Dropping {np.sum(np.isnan(snr_list))} NaN values from SNR array")
    pr_snr_list = snr_list[~np.isnan(snr_list)]
    print(f"Dropping {np.sum(np.isnan(psnr_list))} NaN values from PSNR array")
    pr_psnr_list = psnr_list[~np.isnan(psnr_list)]
    print(f"Dropping {np.sum(np.isnan(ssim_list))} NaN values from SSIM array")
    pr_ssim_list = ssim_list[~np.isnan(ssim_list)]
    print("====================================")
    print("Performance Metrics:")
    print(f"MSE: {np.mean(pr_mse_list)} ± {np.std(pr_mse_list)}")
    print(f"SNR: {np.mean(pr_snr_list)} ± {np.std(pr_snr_list)}")
    print(f"PSNR: {np.mean(pr_psnr_list)} ± {np.std(pr_psnr_list)}")
    print(f"SSIM: {np.mean(pr_ssim_list)} ± {np.std(pr_ssim_list)}")
    print("====================================")
    print("Quantiles of Performance Metrics:")
    for quantile in [0.25, 0.5, 0.75]:
        print(f"MSE {quantile}: {np.quantile(pr_mse_list, quantile)}")
        print(f"SNR {quantile}: {np.quantile(pr_snr_list, quantile)}")
        print(f"PSNR {quantile}: {np.quantile(pr_psnr_list, quantile)}")
        print(f"SSIM {quantile}: {np.quantile(pr_ssim_list, quantile)}")
    print("====================================")

    performance_metrics = {
        "Subject Name": subject_names,
        "Slice Index": slice_indices,
        "MSE": mse_list,
        "SNR": snr_list,
        "PSNR": psnr_list,
        "SSIM": ssim_list
    }
    performance_metrics_df = pd.DataFrame(performance_metrics)
    checkpoint_fname = os.path.basename(args.pre_train).replace('.pt', '')
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        store_fpath = os.path.join(args.save_dir, f"performance_metrics_{checkpoint_fname}.xlsx")
        print(f"Saving performance metrics to '{store_fpath}' ...")
        performance_metrics_df.to_excel(store_fpath)


if __name__ == "__main__":
    if torch.cuda.is_available():
        main(args, use_gpu=True)
    else:
        main(args, use_gpu=False)
