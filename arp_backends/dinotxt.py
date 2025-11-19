import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DinoTxtWrapper(nn.Module):
    """
    Exact reproduction of the official dinotxt.ipynb behavior.
    - Loads the real DinoTXT model (not via torch.hub!).
    - Uses get_visual_class_and_patch_tokens() from DinoTXT.
    - Uses correct half of text embedding.
    - Applies DinoTXT preprocessing exactly.
    """

    def __init__(self, isize=518):
        super().__init__()

        # ----------------------------------------------------------
        # 1. Load official DinoTXT model (correct way!)
        # ----------------------------------------------------------
        try:
            from dinov2.hub.dinotxt import (
                dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
                get_tokenizer,
            )
        except Exception as e:
            raise RuntimeError(
                "dinov2 is not installed correctly. "
                "Clone the official repo and pip install -e ."
            ) from e

        self.model = (
            dinov2_vitl14_reg4_dinotxt_tet1280d20h24l(pretrained=True)
            .eval()
            .to(DEVICE)
        )

        self.tokenizer = get_tokenizer()

        # ----------------------------------------------------------
        # 2. DinoTXT image preprocess (identical to notebook)
        # ----------------------------------------------------------
        self.preprocess = transforms.Compose([
            transforms.Resize(isize, antialias=True),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        self.isize = isize

    # --------------------------------------------------------------
    # Tokenizer wrapper
    # --------------------------------------------------------------
    def _tok(self, texts):
        ids = [torch.tensor(self.tokenizer.encode(t), dtype=torch.long)
               for t in texts]
        L = max(x.numel() for x in ids)
        batch = torch.zeros(len(ids), L, dtype=torch.long)
        for i, seq in enumerate(ids):
            batch[i, :seq.numel()] = seq
        return batch.to(DEVICE)

    # --------------------------------------------------------------
    # Text encoder (correct half)
    # --------------------------------------------------------------
    @torch.no_grad()
    def encode_text_local(self, texts):
        """
        DinoTXT returns 2D text embedding; second half aligns with patch tokens.
        """
        t2 = self.model.encode_text(self._tok(texts))  # [B, 2D]
        D = t2.shape[-1] // 2
        return t2[:, D:]  # unnormalized

    # --------------------------------------------------------------
    # Patch tokens (the correct DinoTXT ones)
    # --------------------------------------------------------------
    @torch.no_grad()
    def encode_patch_tokens(self, images):
        """
        EXACT DinoTXT behavior:
        get_visual_class_and_patch_tokens returns:
          class_tokens : [B, D]
          patch_tokens : [B, P, D]
        """
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(DEVICE)
        else:
            x = images.to(DEVICE)

        cls_tok, patch_tok = self.model.get_visual_class_and_patch_tokens(x)
        return patch_tok  # [B, P, D]

def load_dinotxt(device="cuda", isize=518):
    m = DinoTxtWrapper(isize=isize)
    return m, m.preprocess

# ----------------------------------------------------------------------
# FINAL: interpret function, exactly as notebook
# ----------------------------------------------------------------------
@torch.no_grad()
def interpret_dinotxt(images, texts, model: DinoTxtWrapper,
                      upsample_size=None):
    """
    Compute DinoTXT heatmaps.
    - images: [B,3,H,W] preprocessed by model.preprocess
    - texts : list[str]
    - upsample_size : (H,W) optional, recommended to use original image size
    """

    device = next(model.parameters()).device

    patches = model.encode_patch_tokens(images)   # [B,P,D]
    B, P, D = patches.shape
    
    H = W = int(P**0.5)

    # reshape tokens to 2D feature map
    x = patches.movedim(2, 1).unflatten(2, (H, W)).float()  # [B,D,H,W]

    # upsample to desired size
    if upsample_size is None:
        # default: return square HxH (not recommended)
        up_h, up_w = H, H
    else:
        up_h, up_w = upsample_size

    x = F.interpolate(
        x,
        size=(up_h, up_w),
        mode="bicubic",
        align_corners=False,
    )

    # normalize patch features
    x = F.normalize(x, p=2, dim=1)

    # text embeddings
    t = model.encode_text_local(texts)
    t = F.normalize(t, p=2, dim=1)

    # cosine similarity
    heatmaps = torch.einsum("bdhw,bd->bhw", x, t).unsqueeze(1)

    # normalize to [0,1]
    hmin = heatmaps.amin(dim=(2, 3), keepdim=True)
    hmax = heatmaps.amax(dim=(2, 3), keepdim=True)
    heatmaps = (heatmaps - hmin) / (hmax - hmin + 1e-6)

    return heatmaps   # [B,1,H,W]
