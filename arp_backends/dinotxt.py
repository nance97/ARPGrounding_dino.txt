import os, math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DinoTxtWrapper(torch.nn.Module):
    def __init__(self, hub_variant="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l", isize=518, use_half=True):
        super().__init__()
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
        m = torch.hub.load("facebookresearch/dinov2", hub_variant, trust_repo=True, force_reload=False).eval()
        m = m.to(DEVICE).half() if (DEVICE == "cuda" and use_half) else m.to(DEVICE)
        self.model = m
        from dinov2.hub.dinotxt import get_tokenizer
        self.tokenizer = get_tokenizer()
        self.preprocess = transforms.Compose([
            transforms.Resize(isize, antialias=True),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    def _tok(self, texts):
        ids = [torch.tensor(self.tokenizer.encode(t), dtype=torch.long) for t in texts]
        L = max(x.numel() for x in ids)
        batch = torch.zeros(len(ids), L, dtype=torch.long)
        for i, seq in enumerate(ids):
            batch[i, :seq.numel()] = seq
        return batch.to(DEVICE)

    @torch.no_grad()
    def encode_text_local(self, texts):
        T2 = self.model.encode_text(self._tok(texts))  # [B, 2D]
        D = T2.shape[-1] // 2
        t = T2[:, D:]                                   # local half
        return F.normalize(t, dim=-1)

    @torch.no_grad()
    def encode_patches(self, images):
        # images: tensor [B,3,H,W] or list[PIL]
        if isinstance(images, list) and len(images) and isinstance(images[0], Image.Image):
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(DEVICE)
        else:
            x = images.to(DEVICE)
        _cls, patches = self.model.get_visual_class_and_patch_tokens(x)  # [B,P,D]
        return F.normalize(patches, dim=-1)  # [B,P,D]

def load_dinotxt(device="cuda", isize=518):
    m = DinoTxtWrapper(isize=isize)
    return m, m.preprocess

@torch.no_grad()
def interpret_dinotxt(images, texts, model, isize=518, device="cuda"):
    """
    images: [B,3,H,W] tensor on device
    texts:  list[str]
    returns heatmaps [B,1,H,W] in [0,1], like interpret_clip
    """
    patches = model.encode_patches(images)          # [B,P,D]
    text_emb = model.encode_text_local(texts)       # [B,D]
    B, P, D = patches.shape
    g = int(math.sqrt(P))
    patches = patches[:, : g*g, :]                  # [B,g*g,D]
    patches = patches.view(B, g*g, D)
    # cosine sim per patch: [B,g*g]
    sims = torch.einsum("bpd,bd->bp", patches, text_emb)
    sims = sims.view(B, 1, g, g)
    sims = (sims - sims.amin(dim=(2,3), keepdim=True)) / (sims.amax(dim=(2,3), keepdim=True) - sims.amin(dim=(2,3), keepdim=True) + 1e-6)
    sims = F.interpolate(sims, size=(isize, isize), mode="bilinear", align_corners=False)
    return sims  # [B,1,H,W] in [0,1]
