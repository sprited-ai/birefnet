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
from PIL import Image, ImageFilter
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
    "toonout":      (None,                              1024),  # anime/stylized fine-tune
    "general":      ("ZhengPeng7/BiRefNet",             1024),  # general-purpose (default)
    "general-hr":   ("ZhengPeng7/BiRefNet_HR",          2048),  # high-resolution general
    "portrait":     ("ZhengPeng7/BiRefNet-portrait",    1024),  # human portraits
    "matting":      ("ZhengPeng7/BiRefNet-matting",     1024),  # trimap-free soft matting
    "matting-hr":   ("ZhengPeng7/BiRefNet_HR-matting",  2048),  # high-res matting
    "dynamic":      ("ZhengPeng7/BiRefNet_dynamic",     1024),  # robust across resolutions
    "lite":         ("ZhengPeng7/BiRefNet_lite",        1024),  # Swin-Tiny, faster/cheaper
    "lite-2k":      ("ZhengPeng7/BiRefNet_lite-2K",     2048),  # Swin-Tiny at 2K
}

DEFAULT_VARIANT = "general"

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
        # Warm container keeps every touched variant resident (Swin-Large is a
        # few GB in fp16, comfortable on an A100-80GB). Preload the baked
        # variants so their first request is instant; the rest lazy-load.
        self._models: dict[str, AutoModelForImageSegmentation] = {}
        self._load("toonout")
        self._load(DEFAULT_VARIANT)

    def _load(self, variant: str) -> AutoModelForImageSegmentation:
        """Lazily build + cache the model for a variant."""
        if variant in self._models:
            return self._models[variant]

        repo, _ = VARIANTS[variant]
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
        if self.device == "cuda":
            model.half()
        self._models[variant] = model
        return model

    def predict(
        self,
        image: Path = Input(description="Input image"),
        variant: str = Input(
            description="Which BiRefNet model to use. 'general' is the "
            "all-purpose model; 'toonout' is the anime/stylized fine-tune; "
            "the rest are specialized BiRefNet zoo models.",
            choices=list(VARIANTS.keys()),
            default=DEFAULT_VARIANT,
        ),
        resolution: int = Input(
            description="Inference resolution (square). 0 = use the variant's "
            "native resolution (1024, or 2048 for HR/2K variants).",
            default=0, ge=0, le=2048,
        ),
        output_format: str = Input(
            description="'cutout' = RGBA image with background removed; "
            "'mask' = the raw single-channel alpha matte.",
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
            description="Refine foreground colours (FB blur fusion) to remove "
            "background bleed on soft edges. Ignored for 'mask' output.",
            default=False,
        ),
    ) -> Path:
        model = self._load(variant)
        _, native_res = VARIANTS[variant]
        res = resolution or native_res

        src = Image.open(str(image)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        batch = tf(src).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            batch = batch.half()

        with torch.no_grad():
            preds = model(batch)[-1].sigmoid().float().cpu()
        matte = transforms.ToPILImage()(preds[0].squeeze()).resize(src.size)

        if mask_offset > 0:
            for _ in range(mask_offset):
                matte = matte.filter(ImageFilter.MaxFilter(3))
        elif mask_offset < 0:
            for _ in range(-mask_offset):
                matte = matte.filter(ImageFilter.MinFilter(3))
        if mask_blur > 0:
            matte = matte.filter(ImageFilter.GaussianBlur(mask_blur))

        out = pathlib.Path("/tmp/output.png")
        if output_format == "mask":
            matte.save(out)
            return Path(out)

        if refine_fg:
            src = refine_foreground(src, matte)
        rgba = src.convert("RGBA")
        rgba.putalpha(matte)
        rgba.save(out)
        return Path(out)
