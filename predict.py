"""BiRefNet background removal on Replicate — the full variant zoo.

One predictor, every BiRefNet variant, picked per request via `variant`.
All variants share the same network, so a single warm container serves them
all. Includes **ToonOut** (arXiv:2509.06839, MIT) — a BiRefNet fine-tune by
Muratori & Seytre specialized for stylized/anime content (hair wisps, line
art, translucency; pixel accuracy 95.3% -> 99.5% on their anime test set) —
alongside the official BiRefNet model zoo (ZhengPeng7).

For the dedicated anime-only endpoint, see sprited/birefnet-toonout.

Input: any image. Output: RGBA cutout (or raw matte).
"""

import pathlib

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image, ImageFilter, ImageSequence
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Baked into the image at build time (see cog.yaml). ToonOut ships as a bare
# state_dict, so it loads onto the stock BiRefNet architecture.
TOONOUT_WEIGHTS = "/weights/birefnet_finetuned_toonout.pth"
TOONOUT_BASE_ARCH = "ZhengPeng7/BiRefNet"

# variant -> (huggingface repo | None for toonout, native square resolution).
# Native res is the training resolution used when `resolution=0` (auto).
# All variants share the BiRefNet architecture; the official repos carry
# their own arch config via trust_remote_code (so _lite picks Swin-Tiny and
# the HR/2K repos run at 2048 automatically).
VARIANTS: dict[str, tuple[str | None, int]] = {
    "general":      ("ZhengPeng7/BiRefNet",             1024),  # general-purpose (default)
    "general-hr":   ("ZhengPeng7/BiRefNet_HR",          2048),  # high-resolution general
    "portrait":     ("ZhengPeng7/BiRefNet-portrait",    1024),  # human portraits
    "matting":      ("ZhengPeng7/BiRefNet-matting",     1024),  # trimap-free soft matting
    "matting-hr":   ("ZhengPeng7/BiRefNet_HR-matting",  2048),  # high-res matting
    "dynamic":      ("ZhengPeng7/BiRefNet_dynamic",     1024),  # robust across resolutions
    "dynamic-matting": ("ZhengPeng7/BiRefNet_dynamic-matting", 1024),  # dynamic-res soft matting
    "lite":         ("ZhengPeng7/BiRefNet_lite",        1024),  # Swin-Tiny, faster/cheaper
    "lite-2k":      ("ZhengPeng7/BiRefNet_lite-2K",     2048),  # Swin-Tiny at 2K
    "lite-matting": ("ZhengPeng7/BiRefNet_lite-matting", 1024),  # Swin-Tiny soft matting
    "toonout":      (None,                              1024),  # anime/stylized fine-tune
}

DEFAULT_VARIANT = "general"

# Cap animated inputs so a giant GIF can't tie up the container indefinitely.
MAX_ANIMATED_FRAMES = 300

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def _fb_blur_fusion(image, F, B, alpha, r):
    # Approximate Fast Foreground Colour Estimation (Germer et al. 2020),
    # same implementation as BiRefNet's image_proc.refine_foreground
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
    blurred_F = cv2.blur(F * alpha, (r, r)) / (blurred_alpha + 1e-5)
    blurred_B = cv2.blur(B * (1 - alpha), (r, r)) / (1 - blurred_alpha + 1e-5)
    F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    return np.clip(F, 0, 1), blurred_B


def refine_foreground(image: Image.Image, matte: Image.Image, r: int = 90) -> Image.Image:
    """Strip background colour bleed from semi-transparent edge pixels."""
    img = np.asarray(image, dtype=np.float32) / 255.0
    alpha = (np.asarray(matte, dtype=np.float32) / 255.0)[:, :, None]
    F, blurred_B = _fb_blur_fusion(img, img, img, alpha, r)
    F, _ = _fb_blur_fusion(img, F, blurred_B, alpha, 6)
    return Image.fromarray((F * 255.0 + 0.5).astype(np.uint8))


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[setup] device={self.device}, preloading baked variants", flush=True)
        # Warm container keeps every touched variant resident (Swin-Large is a
        # few GB in fp16, comfortable on an A100-80GB). Preload the baked
        # variants so their first request is instant; the rest lazy-load.
        self._models: dict[tuple[str, str], AutoModelForImageSegmentation] = {}
        self._load("toonout", "fp32")
        self._load(DEFAULT_VARIANT, "fp32")
        print("[setup] ready", flush=True)

    def _load(self, variant: str, precision: str) -> AutoModelForImageSegmentation:
        """Lazily build + cache the model for a (variant, precision) pair."""
        key = (variant, precision)
        if key in self._models:
            return self._models[key]

        repo, _ = VARIANTS[variant]
        print(f"[load] building '{variant}' [{precision}] ({repo or 'toonout .pth'}) — "
              "first use of a non-baked variant downloads its weights",
              flush=True)
        if repo is None:  # toonout: stock arch + baked fine-tune weights
            model = AutoModelForImageSegmentation.from_pretrained(
                TOONOUT_BASE_ARCH, trust_remote_code=True
            )
            state = torch.load(TOONOUT_WEIGHTS, map_location="cpu", weights_only=True)
            # checkpoint was saved from a DDP + torch.compile wrapper
            state = {k.removeprefix("module.").removeprefix("_orig_mod."): v
                     for k, v in state.items()}
            model.load_state_dict(state)
        else:  # official zoo: repo carries its own arch + weights
            model = AutoModelForImageSegmentation.from_pretrained(
                repo, trust_remote_code=True
            )

        model.to(self.device).eval()
        if self.device == "cuda" and precision == "fp16":
            model.half()
        self._models[key] = model
        return model

    def _matte_frame(self, model, rgb: Image.Image, res: int, output_format: str,
                     mask_blur: int, mask_offset: int, refine_fg: bool, half: bool) -> Image.Image:
        """Run BiRefNet on one RGB frame and return the cutout (RGBA) or matte (L)."""
        tf = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        batch = tf(rgb).unsqueeze(0).to(self.device)
        if half:
            batch = batch.half()

        with torch.no_grad():
            preds = model(batch)[-1].sigmoid().float().cpu()
        matte = transforms.ToPILImage()(preds[0].squeeze()).resize(rgb.size)

        if mask_offset > 0:
            for _ in range(mask_offset):
                matte = matte.filter(ImageFilter.MaxFilter(3))
        elif mask_offset < 0:
            for _ in range(-mask_offset):
                matte = matte.filter(ImageFilter.MinFilter(3))
        if mask_blur > 0:
            matte = matte.filter(ImageFilter.GaussianBlur(mask_blur))

        if output_format == "mask":
            return matte
        fg = refine_foreground(rgb, matte) if refine_fg else rgb
        rgba = fg.convert("RGBA")
        rgba.putalpha(matte)
        return rgba

    def predict(
        self,
        image: Path = Input(description="Input image. Animated GIF/WebP are supported — every frame is matted and returned as an animated WebP with transparency."),
        variant: str = Input(
            description="Which BiRefNet model to use. 'general' is the all-purpose default; 'toonout' is the anime/stylized fine-tune; the rest are specialized BiRefNet zoo models.",
            choices=list(VARIANTS.keys()),
            default=DEFAULT_VARIANT,
        ),
        resolution: int = Input(
            description="Inference resolution (square). 0 = use the variant's native resolution (1024, or 2048 for HR/2K variants).",
            default=0, ge=0, le=2048,
        ),
        output_format: str = Input(
            description="'cutout' = RGBA image with background removed; 'mask' = the raw single-channel alpha matte.",
            choices=["cutout", "mask"],
            default="cutout",
        ),
        mask_blur: int = Input(
            description="Gaussian blur radius applied to the matte (softer edges)",
            default=0, ge=0, le=64,
        ),
        mask_offset: int = Input(
            description="Grow (+) or shrink (-) the matte by this many pixels",
            default=0, ge=-64, le=64,
        ),
        refine_fg: bool = Input(
            description="Refine foreground colours (FB blur fusion) to remove background bleed on soft edges. Ignored for 'mask' output.",
            default=False,
        ),
        precision: str = Input(
            description="GPU inference precision. 'fp32' (default) is full precision. 'fp16' is somewhat faster and uses less VRAM (modest on A100 — ~87ms vs ~69ms; bigger gains on older GPUs), with negligible quality difference.",
            choices=["fp16", "fp32"],
            default="fp32",
        ),
    ) -> Path:
        half = precision == "fp16" and self.device == "cuda"
        model = self._load(variant, precision)
        _, native_res = VARIANTS[variant]
        res = resolution or native_res
        print(f"[predict] variant={variant} resolution={res} precision={precision} output={output_format}", flush=True)

        src = Image.open(str(image))
        animated = getattr(src, "is_animated", False) and getattr(src, "n_frames", 1) > 1

        if not animated:
            frame = self._matte_frame(model, src.convert("RGB"), res,
                                      output_format, mask_blur, mask_offset, refine_fg, half)
            out = pathlib.Path("/tmp/output.png")
            frame.save(out)
            print("[predict] done (still image)", flush=True)
            return Path(out)

        # Animated GIF / WebP: matte every frame and re-encode as an animated
        # WebP — full alpha, unlike GIF's 1-bit transparency. Timing + loop kept.
        n = src.n_frames
        if n > MAX_ANIMATED_FRAMES:
            raise ValueError(
                f"Animated input has {n} frames; the limit is {MAX_ANIMATED_FRAMES}. "
                "Trim or split the animation into shorter clips."
            )
        print(f"[predict] animated input — matting {n} frames", flush=True)
        frames, durations = [], []
        for i, frame in enumerate(ImageSequence.Iterator(src)):
            durations.append(frame.info.get("duration", 100))
            matted = self._matte_frame(model, frame.convert("RGB"), res,
                                       output_format, mask_blur, mask_offset, refine_fg, half)
            # WebP wants RGB/RGBA; the 'mask' branch yields an 'L' frame.
            frames.append(matted if matted.mode == "RGBA" else matted.convert("RGB"))
            print(f"[predict] frame {i + 1}/{n}", flush=True)

        out = pathlib.Path("/tmp/output.webp")
        print(f"[predict] encoding animated WebP ({len(frames)} frames)", flush=True)
        frames[0].save(out, format="WEBP", save_all=True, append_images=frames[1:],
                       duration=durations, loop=src.info.get("loop", 0),
                       lossless=True, method=4)
        print("[predict] done (animated)", flush=True)
        return Path(out)
