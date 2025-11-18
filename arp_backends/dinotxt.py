import os, math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DinoTxtWrapper(torch.nn.Module):
    """
    Strictly reproduces the official dinotxt notebook behavior:
    - uses get_visual_class_and_patch_tokens
    - uses local half of text embedding (un-normalized)
    - normalization happens ONLY once in the heatmap function
    """
    def __init__(self, hub_variant="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
                 isize=518):
        super().__init__()
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))

        m = torch.hub.load(
            "facebookresearch/dinov2",
            hub_variant,
            trust_repo=True,
            force_reload=False,
        ).eval().to(DEVICE)
        self.model = m

        # tokenizer from dinotxt implementation
        from dinov2.hub.dinotxt import get_tokenizer
        self.tokenizer = get_tokenizer()

        # image pipeline matches Meta notebook exactly
        self.preprocess = transforms.Compose([
            transforms.Resize(isize, antialias=True),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.isize = isize

    def _tok(self, texts):
        ids = [torch.tensor(self.tokenizer.encode(t), dtype=torch.long)
               for t in texts]
        L = max(x.numel() for x in ids)
        batch = torch.zeros(len(ids), L, dtype=torch.long)
        for i, seq in enumerate(ids):
            batch[i, :seq.numel()] = seq
        return batch.to(DEVICE)

    @torch.no_grad()
    def encode_text_local(self, texts):
        """
        EXACT DINOTXT BEHAVIOR:
        encode_text -> take second half -> DO NOT normalize here.
        """
        T2 = self.model.encode_text(self._tok(texts))  # [B, 2D]
        D = T2.shape[-1] // 2
        return T2[:, D:]  # [B,D], unnormalized

    @torch.no_grad()
    def encode_patch_tokens(self, images):
        """
        EXACT DINOTXT BEHAVIOR:
        get_visual_class_and_patch_tokens -> return patch tokens only
        (unnormalized at this point; notebook normalizes later)
        """
        if isinstance(images, list) and images and isinstance(images[0], Image.Image):
            x = torch.stack([self.preprocess(im.convert("RGB"))
                             for im in images]).to(DEVICE)
        else:
            x = images.to(DEVICE)

        _cls, patches = self.model.get_visual_class_and_patch_tokens(x)
        return patches  # [B,P,D], unnormalized
    
def load_dinotxt(device="cuda", isize=518):
    m = DinoTxtWrapper(isize=isize)
    return m, m.preprocess

@torch.no_grad()
def interpret_dinotxt(images, texts, model: DinoTxtWrapper,
                      isize=518, device="cuda"):
    """
    More memory-efficient version:
    - images: [1, 3, H, W]
    - texts: list[str] (length T)
    Returns: [T, 1, isize, isize]
    """
    images = images.to(device)
    assert images.size(0) == 1, "Expect a single image per call"

    # Patch tokens: [1, P, D]
    patches = model.encode_patch_tokens(images)
    _, P, D = patches.shape

    H = W = int(P ** 0.5)
    x = patches.movedim(2, 1).unflatten(2, (H, W)).float()  # [1, D, H, W]

    x = F.interpolate(x, size=(isize, isize),
                      mode="bicubic", align_corners=False)
    x = F.normalize(x, p=2, dim=1)  # [1, D, H, W]

    # Texts: [T, D]
    t = model.encode_text_local(texts)
    t = F.normalize(t, p=2, dim=1)  # [T, D]

    # Similarity: [1, T, H, W]
    sims = torch.einsum("bdhw,td->bthw", x, t)[0]  # [T, H, W]

    # Min–max normalize per text
    sims_min = sims.amin(dim=(1, 2), keepdim=True)
    sims_max = sims.amax(dim=(1, 2), keepdim=True)
    sims = (sims - sims_min) / (sims_max - sims_min + 1e-6)

    return sims.unsqueeze(1)  # [T, 1, H, W]
