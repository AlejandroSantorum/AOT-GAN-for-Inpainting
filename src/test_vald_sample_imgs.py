import importlib
import os
import re
import random

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

from utils.option import args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args, use_gpu=True):
    brain_input_dir = os.path.join(args.dir_image, args.data_train)
    mask_input_dir = os.path.join(args.dir_mask, args.mask_type)

    # Define test subjects
    if args.data_train.startswith("IXI-"):
        if args.split_type == "train":
            raise ValueError("Train split for IXI dataset cannot be used in testing phase.")
        elif args.split_type == "test":
            test_subjects = [
                # subject name, brain slice name
                ("IXI296-HH-1970-T1", "IXI296-HH-1970-T1-brain-slice113"),
                ("IXI465-HH-2176-T1", "IXI465-HH-2176-T1-brain-slice120"),
                ("IXI593-Guys-1109-T1", "IXI593-Guys-1109-T1-brain-slice141"),
            ]
        else:
            raise ValueError(f"Unknown split type '{args.split_type}' for IXI dataset. Valid option: 'test'.")
    elif args.data_train.startswith("openneuro-ds000228-"):
        if args.split_type == "train":
            raise ValueError("Train split for OpenNeuro 228 dataset cannot be used in testing phase.")
        elif args.split_type == "test":
            raise NotImplementedError("Test split for OpenNeuro 228 dataset is not implemented yet.")
        else:
            raise ValueError(f"Unknown split type '{args.split_type}' for OpenNeuro 228 dataset. Valid option is: 'test1'.")
    elif args.data_train.startswith("bratsc2023-"):
        if args.split_type == "train":
            raise ValueError("Train split for BraTSC 2023 dataset cannot be used in testing phase.")
        elif args.split_type == "test":
            test_subjects = [
                # subject name, brain slice name
                ("BraTS-GLI-00739-000", "BraTS-GLI-00739-000-slice100"),
                ("BraTS-GLI-01199-000", "BraTS-GLI-01199-000-slice125"),
            ]
        else:
            raise ValueError(f"Unknown split type '{args.split_type}' for BraTSC 2023 dataset. Valid option: 'test'.")    
    else:
        raise ValueError(f"Unknown dataset '{args.data_train}'. Valid options contain: 'IXI', 'openneuro-ds000228', 'bratsc2023'.")

    # Create output directories if they do not exist
    if args.npy_output_dir:
        os.makedirs(os.path.join(args.npy_output_dir, "inpainted"), exist_ok=True)
        os.makedirs(os.path.join(args.npy_output_dir, "ref_mask"), exist_ok=True)
        os.makedirs(os.path.join(args.npy_output_dir, "groundtruth"), exist_ok=True)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

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

    # Set seed for reproducibility
    if args.seed:
        set_seed(args.seed)

    # Inpainting loop for validation subjects
    for subject_name, brain_slice_name in tqdm(test_subjects):
        brain_slice_fpath = os.path.join(brain_input_dir, subject_name, f"{brain_slice_name}.png")
        mask_slice_name = brain_slice_name.replace("-brain-", "-mask-")
        mask_slice_fpath = os.path.join(mask_input_dir, subject_name, f"{mask_slice_name}.png")

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
        groundtruth_npy = brain_slice_img[0].cpu().numpy()[0]
        mask_npy = mask_slice_img[0].cpu().numpy()[0]
        pred_npy = pred_img[0].cpu().numpy()[0]

        # Store slices
        if args.npy_output_dir:
            np.save(
                file=os.path.join(args.npy_output_dir, "inpainted", f"{brain_slice_name}.npy"),
                arr=pred_npy,
            )
            np.save(
                file=os.path.join(args.npy_output_dir, "ref_mask", f"{brain_slice_name}.npy"),
                arr=mask_npy,
            )
            np.save(
                file=os.path.join(args.npy_output_dir, "groundtruth", f"{brain_slice_name}.npy"),
                arr=groundtruth_npy,
            )


if __name__ == "__main__":
    if torch.cuda.is_available():
        main(args, use_gpu=True)
    else:
        main(args, use_gpu=False)
