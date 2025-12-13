import torch
import torch.nn.functional as F

from dinov2.hub.dinotxt import (
    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
    get_tokenizer,
)
from dinov2.data.transforms import make_classification_eval_transform

class DinoTxtBackend:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model + tokenizer exactly as in the notebook
        self.model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(device)
        self.model.eval()

        self.tokenizer = get_tokenizer()
        # Same eval transform as the notebook (if you want to use it on raw PIL imgs)
        self.preprocess = make_classification_eval_transform()

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    def _forward_tokens(self, images, texts):
        """
        images: [B, 3, H, W] tensor on self.device
        texts:  list[str], len = B

        returns:
            image_patch_tokens: [B, P, D]
            text_patch_features: [B, D]  (patch-aligned half of text embedding)
        """
        # Tokenize texts exactly like in the notebook
        tokenized = self.tokenizer.tokenize(texts).to(self.device)  # [B, L]

        # Visual tokens
        image_class_tokens, image_patch_tokens = self.model.get_visual_class_and_patch_tokens(
            images
        )  # [B, D], [B, P, D]

        # Text embedding, then take the patch-aligned half
        text_features = self.model.encode_text(tokenized)  # [B, 2D]
        B, twoD = text_features.shape
        D = twoD // 2
        text_patch_features = text_features[:, D:]  # [B, D]  (same as [:, 1024:] for ViT-L)

        return image_patch_tokens, text_patch_features

    @torch.no_grad()
    def get_heatmaps(self, images, texts):
        """
        images: [B, 3, H, W]  -- this is what ARPGrounding gives you
        texts:  list[str] of length B

        returns:
            heatmaps: [B, 1, H, W]  (values ~ cosine similarity)
        """
        images = images.to(self.device)

        B, _, H, W = images.shape

        # 1) get patch tokens + patch-aligned text features using official APIs
        image_patch_tokens, text_patch_features = self._forward_tokens(images, texts)
        # image_patch_tokens: [B, P, D]
        # text_patch_features: [B, D]

        B2, P, D = image_patch_tokens.shape
        assert B2 == B, "Batch mismatch"
        H_p = W_p = int(P ** 0.5)
        assert H_p * W_p == P, f"Cannot reshape P={P} into square grid"

        # 2) [B, P, D] -> [B, D, H_p, W_p]
        x = image_patch_tokens.movedim(2, 1).unflatten(2, (H_p, W_p)).float()  # [B, D, H_p, W_p]

        # 3) upsample to image resolution
        x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)  # [B, D, H, W]

        # 4) L2-normalize both sides
        x = F.normalize(x, p=2, dim=1)                                # [B, D, H, W]
        y = F.normalize(text_patch_features.float(), p=2, dim=1)      # [B, D]

        # 5) cosine sim per pixel: [B, H, W]
        # Note: in the notebook they do "bdhw,cd->bchw" to handle multiple texts.
        # Here we have 1 text per image → use "bdhw,bd->bhw"
        sims = torch.einsum("bdhw,bd->bhw", x, y)  # [B, H, W]

        # 6) match ARPGrounding's expected shape [B, 1, H, W]
        heatmaps = sims.unsqueeze(1)  # [B, 1, H, W]

        # Optional: ReLU + per-image max-normalization
        heatmaps = torch.relu(heatmaps)
        flat = heatmaps.reshape(B, -1)
        max_vals = flat.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        heatmaps = heatmaps / max_vals.view(B, 1, 1, 1)

        return heatmaps  # [B, 1, H, W]


def load_dinotxt_backend(device: str = None):
    return DinoTxtBackend(device=device)