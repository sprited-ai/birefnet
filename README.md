# birefnet-rmbg

> **Unofficial community packaging.** These are not our models and we are not
> affiliated with their authors. We packaged the
> [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) model zoo (MIT, Peng
> Zheng et al.) and the [ToonOut](https://arxiv.org/abs/2509.06839) fine-tune
> (MIT, Matteo Muratori & Joël Seytre) for
> [Replicate](https://replicate.com); **we earn nothing** — compute fees go to
> Replicate. Authors who want this changed or taken down:
> [open an issue](https://github.com/sprited-ai/birefnet-rmbg/issues) and we
> comply immediately.

**MIT-licensed background removal — commercial-OK, with controls most
one-button removers don't give you.** The full
[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) variant zoo behind one
endpoint, packaged for [Replicate](https://replicate.com).

What sets it apart:

- **Anime / stylized** — the `toonout` variant ([ToonOut](https://arxiv.org/abs/2509.06839))
  handles hair wisps, line art, and translucency that general removers smear.
- **MIT, commercial-free** — BiRefNet and ToonOut are both MIT, so you can
  use the output commercially with no extra license (unlike BRIA-based
  models, which gate commercial use).
- **9 variants, one endpoint** — general / HR / portrait / matting / dynamic /
  lite / 2K / toonout, swapped per request; every variant shares the network
  so one warm container serves them all.
- **More knobs** — choose `cutout` (RGBA) or raw `mask`, grow/shrink/blur the
  matte, refine foreground colours, set inference resolution. Most competitors
  take an image and give you one fixed answer.

For the dedicated anime-only endpoint, see
[sprited/birefnet-toonout](https://replicate.com/sprited/birefnet-toonout).

## Variants

| `variant`    | Model | Best for | Native res |
|--------------|-------|----------|-----------|
| `general`    | BiRefNet | general-purpose (default) | 1024 |
| `general-hr` | BiRefNet_HR | high-resolution photos | 2048 |
| `portrait`   | BiRefNet-portrait | human portraits | 1024 |
| `matting`    | BiRefNet-matting | trimap-free soft matting | 1024 |
| `matting-hr` | BiRefNet_HR-matting | high-res soft matting | 2048 |
| `dynamic`    | BiRefNet_dynamic | robust across resolutions | 1024 |
| `lite`       | BiRefNet_lite | Swin-Tiny, faster/cheaper | 1024 |
| `lite-2k`    | BiRefNet_lite-2K | Swin-Tiny at 2K | 2048 |
| `toonout`    | BiRefNet + ToonOut | anime / stylized content | 1024 |

`general` and `toonout` are baked into the image (instant); the rest
lazy-download from HuggingFace on first use, then stay warm.

## Inputs

- `image` — any image
- `variant` — which model (above), default `general`
- `resolution` — square inference size; `0` = the variant's native res
- `output_format` — `cutout` (RGBA) or `mask` (raw alpha matte)
- `mask_blur` / `mask_offset` — soften / grow / shrink the matte
- `refine_fg` — FB blur-fusion to strip background bleed on soft edges

Packaged by [Sprited](https://spritedx.com).

## Deploy

```
cog login && cog push r8.im/sprited/birefnet-rmbg
```

## Citations

```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
@misc{muratori2025toonout,
  title={ToonOut: Fine-tuned Background Removal for Anime Characters},
  author={Muratori, Matteo and Seytre, Joël},
  year={2025},
  eprint={2509.06839},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
