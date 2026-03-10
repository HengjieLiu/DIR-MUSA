"""
Inference script for DIR-MUSA registration models.

This script:
1) loads a moving/fixed image pair,
2) runs 1-stage or 2-stage registration,
3) saves deformed image + DVF,
4) optionally warps moving segmentation and reports Dice before/after registration.

Run examples:
    # Example 1: loss1 single-stage (with explicit checkpoint)
    python scripts/infer/infer_registration.py \
        --moving-img /path/to/moving_img.nii.gz \
        --fixed-img /path/to/fixed_img.nii.gz \
        --model M01 \
        --loss loss1 \
        --stage 1 \
        --single-stage-resolution r2 \
        --checkpoint-stage1 /path/to/checkpoint_stage1.pth \
        --output-dir /path/to/infer_outputs \
        --output-prefix case001

    # Example 2: loss1 two-stage (stage1+stage2 checkpoints)
    python scripts/infer/infer_registration.py \
        --moving-img /path/to/moving_img.nii.gz \
        --fixed-img /path/to/fixed_img.nii.gz \
        --model M04 \
        --loss loss1 \
        --stage 2 \
        --checkpoint-stage1 /path/to/checkpoint_stage1_r2.pth \
        --checkpoint-stage2 /path/to/checkpoint_stage2_r1.pth \
        --output-dir /path/to/infer_outputs \
        --output-prefix case002

    # Example 3: with segmentations and dice reporting
    python scripts/infer/infer_registration.py \
        --moving-img /path/to/moving_img.nii.gz \
        --fixed-img /path/to/fixed_img.nii.gz \
        --moving-seg /path/to/moving_seg.nii.gz \
        --fixed-seg /path/to/fixed_seg.nii.gz \
        --model M05 \
        --loss loss3 \
        --stage 2 \
        --checkpoint-stage1 /path/to/checkpoint_stage1_r2.pth \
        --checkpoint-stage2 /path/to/checkpoint_stage2_r1.pth \
        --dice-report both \
        --output-dir /path/to/infer_outputs \
        --output-prefix case003 \
        --debug
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import musa


MODEL_MAP = {
    "M01": "01voxelmorph-v1",
    "M04": "04transmorph-v1",
    "M05": "05dualprnet-v1",
}

LOSS_DESC_MAP = {
    "loss1": "similarity only",
    "loss2": "similarity + dice",
    "loss3": "proposed MUSA mode",
}

RESOLUTION_SHAPES = {
    "r1": (160, 160, 192),
    "r2": (80, 80, 96),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DIR-MUSA inference for fixed/moving NIfTI pair. Supports M01/M04/M05 with "
            "loss1/loss2/loss3 settings and 1-stage/2-stage pipelines."
        )
    )

    parser.add_argument("--moving-img", required=True, help="Path to moving image (.nii/.nii.gz)")
    parser.add_argument("--fixed-img", required=True, help="Path to fixed image (.nii/.nii.gz)")
    parser.add_argument("--moving-seg", default=None, help="Optional path to moving segmentation (.nii/.nii.gz)")
    parser.add_argument("--fixed-seg", default=None, help="Optional path to fixed segmentation (.nii/.nii.gz)")

    parser.add_argument(
        "--model",
        required=True,
        choices=["M01", "M04", "M05"],
        help=(
            "Model family. "
            "M01=VoxelMorph, M04=TransMorph, M05=DualPRNet."
        ),
    )
    parser.add_argument(
        "--loss",
        required=True,
        choices=["loss1", "loss2", "loss3"],
        help=(
            "Training recipe to match checkpoints. "
            "loss1=similarity only, loss2=similarity+dice, loss3=proposed MUSA mode. "
            "loss1 supports stage1/stage2; loss2 and loss3 support stage2 only."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        type=int,
        choices=[1, 2],
        help=(
            "Inference pipeline stage count. "
            "1: single-stage model; 2: two-stage pipeline (r2 coarse + r1 fine)."
        ),
    )
    parser.add_argument(
        "--single-stage-resolution",
        default="r1",
        choices=["r1", "r2"],
        help=(
            "Resolution for stage=1 only. "
            "r1: 160x160x192, r2: 80x80x96 (DVF is upsampled back to r1 output space)."
        ),
    )

    parser.add_argument(
        "--checkpoint-stage1",
        default=None,
        help=(
            "Checkpoint path for stage1 model. "
            "Required for stage=1 unless auto-resolved; required for stage=2 unless auto-resolved."
        ),
    )
    parser.add_argument(
        "--checkpoint-stage2",
        default=None,
        help="Checkpoint path for stage2 model (only used for stage=2; auto-resolved if omitted).",
    )
    parser.add_argument(
        "--pretrained-root",
        default=str(PROJECT_ROOT / "pretrained_models"),
        help=(
            "Root folder used for checkpoint auto-discovery when checkpoint path is omitted. "
            "Default: <project_root>/pretrained_models."
        ),
    )

    parser.add_argument("--output-dir", required=True, help="Output folder for saved NIfTI files.")
    parser.add_argument("--output-prefix", default="infer", help="Prefix for output files.")

    parser.add_argument(
        "--dice-report",
        default="both",
        choices=["mean", "per-class", "both"],
        help="Dice report style when segmentations are provided.",
    )
    parser.add_argument(
        "--dice-include-background",
        action="store_true",
        help="Include label 0 in mean Dice. By default label 0 is excluded when present.",
    )

    parser.add_argument("--gpu", default="0", help="CUDA visible GPU IDs (e.g., '0' or '0,1').")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--debug", action="store_true", help="Print array overviews for I/O tensors.")
    return parser.parse_args()


def normalize_combo(args: argparse.Namespace) -> None:
    if args.loss in ("loss2", "loss3") and args.stage != 2:
        raise ValueError(f"{args.loss} only supports 2-stage inference, got --stage {args.stage}.")
    if (args.moving_seg is None) != (args.fixed_seg is None):
        raise ValueError("Both --moving-seg and --fixed-seg must be provided together.")


def print_info(args: argparse.Namespace, device: torch.device) -> None:
    print("\n[INFO] DIR-MUSA inference configuration")
    print(f"[INFO] model={args.model} ({MODEL_MAP[args.model]}), loss={args.loss}, stage={args.stage}")
    print("[INFO] loss definitions: loss1=similarity only; loss2=similarity+dice; loss3=proposed MUSA mode")
    print(f"[INFO] selected loss meaning: {args.loss} -> {LOSS_DESC_MAP[args.loss]}")
    if args.stage == 1:
        print(f"[INFO] single-stage resolution={args.single_stage_resolution}")
    print(f"[INFO] moving_img={args.moving_img}")
    print(f"[INFO] fixed_img={args.fixed_img}")
    print(f"[INFO] moving_seg={args.moving_seg}")
    print(f"[INFO] fixed_seg={args.fixed_seg}")
    print(f"[INFO] pretrained_root={args.pretrained_root}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] output_prefix={args.output_prefix}")
    print(f"[INFO] device={device}")
    print(f"[INFO] debug={args.debug}")


def _ckpt_payload_to_state_dict(payload: dict) -> dict:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _format_count(n: int) -> str:
    return f"{n:,d} ({n / 1e6:.3f} M)"


def _model_param_counts(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _print_model_param_info(name: str, model: torch.nn.Module) -> None:
    total, trainable = _model_param_counts(model)
    print(f"[DEBUG] {name} params: total={_format_count(total)}, trainable={_format_count(trainable)}")


def _score_checkpoint(path: Path, must_tokens: List[str], bonus_tokens: List[str]) -> int:
    text = str(path).lower()
    score = 0
    for token in must_tokens:
        if token in text:
            score += 3
    for token in bonus_tokens:
        if token in text:
            score += 1
    return score


def auto_find_checkpoint(
    pretrained_root: str,
    model_keys: List[str],
    loss_key: str,
    role: str,
) -> Optional[str]:
    root = Path(pretrained_root)
    if not root.is_dir():
        return None

    all_ckpts = list(root.rglob("*.pth"))
    if len(all_ckpts) == 0:
        return None

    loss_token = loss_key.lower()
    model_tokens = [k.lower() for k in model_keys]

    if role == "single_stage":
        must = [loss_token] + model_tokens
        bonus = ["1stage", "single", "stage1"]
    elif role == "stage1":
        must = [loss_token, "2stage"] + model_tokens
        bonus = ["stage1", "coarse", "r2"]
    elif role == "stage2":
        must = [loss_token, "2stage"] + model_tokens
        bonus = ["stage2", "fine", "r1"]
    else:
        raise ValueError(f"Unknown checkpoint role: {role}")

    scored = [(p, _score_checkpoint(p, must, bonus)) for p in all_ckpts]
    scored = [x for x in scored if x[1] > 0]
    if len(scored) == 0:
        return None

    scored.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)
    best = scored[0]
    if len(scored) > 1 and scored[1][1] == best[1]:
        print(f"[WARN] Multiple checkpoint candidates matched role={role}; using latest: {best[0]}")
    return str(best[0])


def load_model(model_type: str, model_resolution: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = musa.utils_model_zoo.get_model_v1(
        inshape=RESOLUTION_SHAPES[model_resolution],
        model_type=model_type,
        model_resolution=model_resolution,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = _ckpt_payload_to_state_dict(payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_single_stage(
    moving_t: torch.Tensor,
    fixed_t: torch.Tensor,
    model: torch.nn.Module,
    model_type: str,
    model_resolution: str,
    spatial_transformer_r1: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    down_scale_cnt = 0 if model_resolution == "r1" else 1
    flag_pad = (model_type.startswith("04transmorph-v1") and model_resolution == "r2")

    moving_low = moving_t
    fixed_low = fixed_t
    for _ in range(down_scale_cnt):
        moving_low = musa.utils_warp.vol_downsamplex2(moving_low)
        fixed_low = musa.utils_warp.vol_downsamplex2(fixed_low)

    inputs = (moving_low, fixed_low)
    if not flag_pad:
        deformed_low, dvf_low = musa.utils_model_zoo.model_register_v1(inputs, model, model_type)
    else:
        pad_size = (16, 16, 24, 24, 24, 24)
        padded_inputs = [F.pad(d, pad=pad_size) for d in inputs]
        deformed_low, dvf_low = musa.utils_model_zoo.model_register_v1(padded_inputs, model, model_type)
        deformed_low = deformed_low[..., 24:24 + 80, 24:24 + 80, 16:16 + 96]
        dvf_low = dvf_low[..., 24:24 + 80, 24:24 + 80, 16:16 + 96]

    dvf_r1 = dvf_low
    for _ in range(down_scale_cnt):
        dvf_r1 = musa.utils_warp.dvf_upsample(dvf_r1)

    deformed_r1 = spatial_transformer_r1(moving_t, dvf_r1, mode="bilinear")
    return deformed_r1, dvf_r1


def run_two_stage(
    moving_t: torch.Tensor,
    fixed_t: torch.Tensor,
    model_stage1: torch.nn.Module,
    model_stage2: torch.nn.Module,
    model_type: str,
    spatial_transformer_r1: torch.nn.Module,
    composer_r1: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flag_pad = model_type.startswith("04transmorph-v1")

    moving_r2 = musa.utils_warp.vol_downsamplex2(moving_t)
    fixed_r2 = musa.utils_warp.vol_downsamplex2(fixed_t)

    inputs_stage1 = (moving_r2, fixed_r2)
    if not flag_pad:
        _, dvf_r2_stage1 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, model_type)
    else:
        pad_size = (16, 16, 24, 24, 24, 24)
        padded_stage1 = [F.pad(d, pad=pad_size) for d in inputs_stage1]
        _, dvf_r2_stage1 = musa.utils_model_zoo.model_register_v1(padded_stage1, model_stage1, model_type)
        dvf_r2_stage1 = dvf_r2_stage1[..., 24:24 + 80, 24:24 + 80, 16:16 + 96]

    dvf_r1_stage1 = musa.utils_warp.dvf_upsample(dvf_r2_stage1)
    deformed_r1_stage1 = spatial_transformer_r1(moving_t, dvf_r1_stage1, mode="bilinear")

    inputs_stage2 = (deformed_r1_stage1, fixed_t)
    _, dvf_r1_stage2 = musa.utils_model_zoo.model_register_v1(inputs_stage2, model_stage2, model_type)

    dvf_composed = composer_r1(dvf_r1_stage1, dvf_r1_stage2)
    deformed_final = spatial_transformer_r1(moving_t, dvf_composed, mode="bilinear")
    return deformed_final, dvf_composed


def remap_to_index(seg_int: np.ndarray, labels_sorted: np.ndarray) -> np.ndarray:
    return np.searchsorted(labels_sorted, seg_int).astype(np.int64)


def to_onehot(seg_int: np.ndarray, labels_sorted: np.ndarray, device: torch.device) -> torch.Tensor:
    seg_idx = remap_to_index(seg_int, labels_sorted)
    seg_idx_t = torch.from_numpy(seg_idx[None, ...]).long().to(device)
    onehot = F.one_hot(seg_idx_t, num_classes=labels_sorted.shape[0]).permute(0, 4, 1, 2, 3).float()
    return onehot


def dice_report(
    moving_seg_int: np.ndarray,
    fixed_seg_int: np.ndarray,
    deformed_seg_int: np.ndarray,
    device: torch.device,
    report_mode: str,
    include_background: bool,
) -> None:
    labels = np.union1d(np.unique(moving_seg_int), np.unique(fixed_seg_int))
    labels = np.union1d(labels, np.unique(deformed_seg_int))
    labels = np.sort(labels.astype(np.int64))

    moving_oh = to_onehot(moving_seg_int, labels, device)
    fixed_oh = to_onehot(fixed_seg_int, labels, device)
    deformed_oh = to_onehot(deformed_seg_int, labels, device)

    dice_before = musa.utils_dice.dice_val(moving_oh, fixed_oh).squeeze(0).detach().cpu().numpy()
    dice_after = musa.utils_dice.dice_val(deformed_oh, fixed_oh).squeeze(0).detach().cpu().numpy()

    valid_mask = np.ones_like(labels, dtype=bool)
    if not include_background and np.any(labels == 0):
        valid_mask = labels != 0

    if np.any(valid_mask):
        mean_before = float(np.mean(dice_before[valid_mask]))
        mean_after = float(np.mean(dice_after[valid_mask]))
    else:
        mean_before = float(np.mean(dice_before))
        mean_after = float(np.mean(dice_after))

    print("\n[INFO] Dice summary")
    print(f"[INFO] labels_count={len(labels)}, include_background={include_background}")
    print(f"[INFO] mean_dice_before={mean_before:.6f}")
    print(f"[INFO] mean_dice_after ={mean_after:.6f}")
    print(f"[INFO] mean_dice_delta ={(mean_after - mean_before):+.6f}")

    if report_mode in ("per-class", "both"):
        print("[INFO] Per-class Dice (label: before -> after)")
        for i, label in enumerate(labels):
            print(f"[INFO] label={int(label):>4d}: {dice_before[i]:.6f} -> {dice_after[i]:.6f}")


def main() -> None:
    args = parse_args()
    normalize_combo(args)

    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    print_info(args, device)
    os.makedirs(args.output_dir, exist_ok=True)

    moving_np, moving_affine = musa.utils_dataloader.load_nifti(args.moving_img, ret_affine=True)
    fixed_np, fixed_affine = musa.utils_dataloader.load_nifti(args.fixed_img, ret_affine=True)
    if moving_np.shape != RESOLUTION_SHAPES["r1"] or fixed_np.shape != RESOLUTION_SHAPES["r1"]:
        raise ValueError(
            "Input image shapes must be r1 (160,160,192) after preprocessing. "
            f"Got moving={moving_np.shape}, fixed={fixed_np.shape}."
        )

    if args.debug:
        musa.utils_basics.numpy_overview(moving_np, "moving_img_in")
        musa.utils_basics.numpy_overview(fixed_np, "fixed_img_in")

    moving_t = musa.utils_basics.numpy2torch(moving_np, device=device, CHECK=True)
    fixed_t = musa.utils_basics.numpy2torch(fixed_np, device=device, CHECK=True)

    spatial_transformer_r1 = musa.utils_warp.SpatialTransformer(RESOLUTION_SHAPES["r1"]).to(device)
    composer_r1 = musa.utils_warp.ComposeDVF(RESOLUTION_SHAPES["r1"]).to(device)

    model_type = MODEL_MAP[args.model]
    model_search_keys = [args.model.lower(), model_type.lower(), model_type.split("-")[0].lower()]

    ckpt_stage1 = args.checkpoint_stage1
    ckpt_stage2 = args.checkpoint_stage2
    if args.stage == 1 and ckpt_stage1 is None:
        ckpt_stage1 = auto_find_checkpoint(args.pretrained_root, model_search_keys, args.loss, "single_stage")
    if args.stage == 2 and ckpt_stage1 is None:
        ckpt_stage1 = auto_find_checkpoint(args.pretrained_root, model_search_keys, args.loss, "stage1")
    if args.stage == 2 and ckpt_stage2 is None:
        ckpt_stage2 = auto_find_checkpoint(args.pretrained_root, model_search_keys, args.loss, "stage2")

    if args.stage == 1 and ckpt_stage1 is None:
        raise ValueError("Cannot resolve stage1 checkpoint automatically. Provide --checkpoint-stage1.")
    if args.stage == 2 and (ckpt_stage1 is None or ckpt_stage2 is None):
        raise ValueError("Cannot resolve 2-stage checkpoints automatically. Provide --checkpoint-stage1 and --checkpoint-stage2.")

    print(f"[INFO] checkpoint_stage1={ckpt_stage1}")
    if args.stage == 2:
        print(f"[INFO] checkpoint_stage2={ckpt_stage2}")

    with torch.no_grad():
        if args.stage == 1:
            model_stage1 = load_model(
                model_type=model_type,
                model_resolution=args.single_stage_resolution,
                checkpoint_path=ckpt_stage1,
                device=device,
            )
            if args.debug:
                _print_model_param_info("stage1 model", model_stage1)
            deformed_t, dvf_t = run_single_stage(
                moving_t=moving_t,
                fixed_t=fixed_t,
                model=model_stage1,
                model_type=model_type,
                model_resolution=args.single_stage_resolution,
                spatial_transformer_r1=spatial_transformer_r1,
            )
        else:
            model_stage1 = load_model(model_type=model_type, model_resolution="r2", checkpoint_path=ckpt_stage1, device=device)
            model_stage2 = load_model(model_type=model_type, model_resolution="r1", checkpoint_path=ckpt_stage2, device=device)
            if args.debug:
                _print_model_param_info("stage1 model", model_stage1)
                _print_model_param_info("stage2 model", model_stage2)
                total_2stage = _model_param_counts(model_stage1)[0] + _model_param_counts(model_stage2)[0]
                print(f"[DEBUG] two-stage total params: total={_format_count(total_2stage)}")
            deformed_t, dvf_t = run_two_stage(
                moving_t=moving_t,
                fixed_t=fixed_t,
                model_stage1=model_stage1,
                model_stage2=model_stage2,
                model_type=model_type,
                spatial_transformer_r1=spatial_transformer_r1,
                composer_r1=composer_r1,
            )

    deformed_np = musa.utils_basics.torch2numpy(deformed_t, CHECK=True)
    dvf_np = musa.utils_basics.torch2numpy(dvf_t, CHECK=True)  # [3, H, W, D]
    dvf_save_np = np.moveaxis(dvf_np, 0, -1)  # [H, W, D, 3]

    if args.debug:
        musa.utils_basics.numpy_overview(deformed_np, "deformed_img_out")
        musa.utils_basics.numpy_overview(dvf_np, "dvf_out_chwd")
        musa.utils_basics.numpy_overview(dvf_save_np, "dvf_out_hwdc")

    out_img = Path(args.output_dir) / f"{args.output_prefix}_deformed_img.nii.gz"
    out_dvf = Path(args.output_dir) / f"{args.output_prefix}_dvf.nii.gz"
    musa.utils_dataloader.save_nifti(deformed_np.astype(np.float32), str(out_img), affine=fixed_affine)
    musa.utils_dataloader.save_nifti(dvf_save_np.astype(np.float32), str(out_dvf), affine=fixed_affine)

    print(f"[INFO] Saved deformed image: {out_img}")
    print(f"[INFO] Saved deformation field: {out_dvf}")

    if args.moving_seg is not None:
        moving_seg_np = musa.utils_dataloader.load_nifti(args.moving_seg)
        fixed_seg_np = musa.utils_dataloader.load_nifti(args.fixed_seg)

        moving_seg_int = np.rint(moving_seg_np).astype(np.int64)
        fixed_seg_int = np.rint(fixed_seg_np).astype(np.int64)

        if args.debug:
            musa.utils_basics.numpy_overview(moving_seg_int, "moving_seg_in")
            musa.utils_basics.numpy_overview(fixed_seg_int, "fixed_seg_in")

        moving_seg_t = musa.utils_basics.numpy2torch(moving_seg_int.astype(np.float32), device=device, CHECK=True)
        deformed_seg_t = spatial_transformer_r1(moving_seg_t, dvf_t, mode="nearest")
        deformed_seg_np = musa.utils_basics.torch2numpy(deformed_seg_t, CHECK=True)
        deformed_seg_int = np.rint(deformed_seg_np).astype(np.int64)

        if args.debug:
            musa.utils_basics.numpy_overview(deformed_seg_int, "deformed_seg_out")

        out_seg = Path(args.output_dir) / f"{args.output_prefix}_deformed_seg.nii.gz"
        musa.utils_dataloader.save_nifti(deformed_seg_int.astype(np.int16), str(out_seg), affine=fixed_affine)
        print(f"[INFO] Saved deformed segmentation: {out_seg}")

        dice_report(
            moving_seg_int=moving_seg_int,
            fixed_seg_int=fixed_seg_int,
            deformed_seg_int=deformed_seg_int,
            device=device,
            report_mode=args.dice_report,
            include_background=args.dice_include_background,
        )
    else:
        print("[INFO] Segmentation not provided, skipping segmentation warping and Dice reporting.")


if __name__ == "__main__":
    main()
