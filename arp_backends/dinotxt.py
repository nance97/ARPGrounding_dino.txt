import os, torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DinoTxtWrapper(torch.nn.Module):
    """
    EXACT replication of the DINO patch/text encoding path.
    Uses forward_features + x_norm_patchtokens + patch_embed.grid_size.
    """
    def __init__(self, hub_variant="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
                 isize=518):
        super().__init__()
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_variant,
            trust_repo=True,
            force_reload=False
        ).eval().to(DEVICE)

        from dinov2.hub.dinotxt import get_tokenizer
        self.tokenizer = get_tokenizer()

        self.preprocess = transforms.Compose([
            transforms.Resize(isize, antialias=True),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406),
                                 std=(0.229,0.224,0.225)),
        ])

    # ---- TEXT ENCODING ----
    def _tok(self, texts):
        ids = [torch.tensor(self.tokenizer.encode(t), dtype=torch.long) for t in texts]
        L = max(x.numel() for x in ids)
        batch = torch.zeros(len(ids), L, dtype=torch.long)
        for i, seq in enumerate(ids):
            batch[i, :seq.numel()] = seq
        return batch.to(DEVICE)

    @torch.no_grad()
    def encode_text_local(self, texts):
        """Strict fidelity: same local-half split and normalization."""
        t2 = self.model.encode_text(self._tok(texts))  # [B, 2D]
        D = t2.shape[-1] // 2
        t_local = t2[:, D:]     # local half
        return torch.nn.functional.normalize(t_local, dim=-1)

    # ---- IMAGE PATCH FEATURES ----
    @torch.no_grad()
    def encode_patch_features(self, images):
        """
        Strict fidelity to DINO TXT:
            - Use forward_features()
            - Use x_norm_patchtokens
            - Use patch_embed.grid_size to reshape
        """
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(DEVICE)
        else:
            x = images.to(DEVICE)

        # EXACT Meta DINO forward
        out = self.model.forward_features(x)

        feats = out["x_norm_patchtokens"]  # [B, HW, D]

        # correct feature resolution (H_feat, W_feat)
        H_feat, W_feat = self.model.patch_embed.grid_size

        B, HW, D = feats.shape
        assert HW == H_feat * W_feat, "Patch count mismatch — strict fidelity requires exact reshape."

        # reshape to [B, D, Hf, Wf]
        fmap = feats.view(B, H_feat, W_feat, D).permute(0, 3, 1, 2)
        fmap = torch.nn.functional.normalize(fmap, dim=1)

        return fmap  # [B, D, Hf, Wf]
    
def load_dinotxt(device="cuda", isize=518):
    m = DinoTxtWrapper(isize=isize)
    return m, m.preprocess

@torch.no_grad()
def interpret_dinotxt(images, texts, model: DinoTxtWrapper,
                             isize=518, device="cuda"):
    """
    EXACT DINO TXT logic:
        1. extract feature grid via forward_features
        2. interpolate features to full resolution
        3. cosine similarity per pixel
        4. min-max normalize
    """

    fmap = model.encode_patch_features(images)

    fmap_up = torch.nn.functional.interpolate(
        fmap, size=(isize, isize), mode="bicubic", align_corners=False
    )  # [B, D, H, W]

    t_local = model.encode_text_local(texts)  # [B, D]
    sim = torch.einsum("bdhw,bd->bhw", fmap_up, t_local).unsqueeze(1)

    sim_min = sim.amin(dim=(2,3), keepdim=True)
    sim_max = sim.amax(dim=(2,3), keepdim=True)
    sim = (sim - sim_min) / (sim_max - sim_min + 1e-6)

    return sim  # [B,1,H,W]
