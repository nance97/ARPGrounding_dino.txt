import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.hub.dinotxt import (
    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
    get_tokenizer,
)
from dinov2.data.transforms import make_classification_eval_transform


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if dim_ff is None:
            dim_ff = 4 * d_model

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,  # [B, L, D]
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,  # query [B, L, D], key/value [B, P, D]
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, txt_tokens, img_tokens, need_attn: bool = False):
        """
        txt_tokens: [B, L, D]
        img_tokens: [B, P, D] (frozen memory)
        """
        # 1) text self-attention
        x = txt_tokens
        sa_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + self.dropout(sa_out)
        x = self.norm1(x)

        # 2) text->image cross-attention
        ca_out, attn = self.cross_attn(
            query=x,
            key=img_tokens,
            value=img_tokens,
            need_weights=need_attn,
            average_attn_weights=False,  # <<--- ADD THIS
        )
        x = x + self.dropout(ca_out)
        x = self.norm2(x)

        # 3) feed-forward
        ff_out = self.ffn(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return (x, attn) if need_attn else (x, None)


class MultiModalEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiModalBlock(d_model, n_heads) for _ in range(num_layers)]
        )
        # ITM head if you later want to train with an image-text matching loss
        self.itm_head = nn.Linear(d_model, 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, txt_tokens, img_tokens, return_attn: bool = False):
        """
        txt_tokens: [B, L, D]
        img_tokens: [B, P, D]
        """
        B, L, D = txt_tokens.shape
        cls = self.cls_token.expand(B, 1, D)        # [B, 1, D]
        x = torch.cat([cls, txt_tokens], dim=1)     # [B, 1+L, D]
        
        last_attn = None
        for i, layer in enumerate(self.layers):
            need_attn = return_attn and (i == len(self.layers) - 1)
            x, attn = layer(x, img_tokens, need_attn=need_attn)
            if attn is not None:
                last_attn = attn  # [B, n_heads, L, P]
        return x, last_attn


class DinoTxtFusionBackend:
    def __init__(self, device: str | None = None, last_k: int = 1,
                 n_heads: int = 8, num_layers: int = 4):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.last_k = last_k

        # --- Load dino.txt model + tokenizer (frozen) ---
        self.model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = get_tokenizer()
        self.preprocess = make_classification_eval_transform()

        # Figure out embedding dim D (patch-aligned half)
        # text_features: [1, 2D] → D = text_features.shape[-1] // 2
        dummy_ids = self.tokenizer.tokenize(["dummy"]).to(device)
        with torch.no_grad():
            tf = self.model.encode_text(dummy_ids)
        D = tf.shape[-1] // 2

        # --- Multimodal fusion encoder (trainable) ---
        self.fusion = MultiModalEncoder(d_model=D, n_heads=n_heads, num_layers=num_layers).to(device)

    # ---------- helpers ----------
    def to(self, device):
        self.device = device
        self.model.to(device)        # dino.txt frozen backbone
        self.fusion.to(device)       # multimodal fusion encoder
        return self

    @torch.no_grad()
    def _get_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns: img_tokens [B, P, D] in patch-aligned space
        """
        images = images.to(self.device)
        _, patch_tokens = self.model.get_visual_class_and_patch_tokens(images)
        # patch_tokens: [B, P, D] already in the correct space
        return patch_tokens

    @torch.no_grad()
    def _get_text_token_batch(self, texts: list[str]) -> torch.Tensor:
        """
        texts: list of length B
        returns: txt_tokens [B, L_max, D] in patch-aligned space, L2-normalized per token.
        We DO NOT drop special tokens here for speed; we can mask them later if needed.
        """
        m = self.model
        tok = self.tokenizer
        dev = self.device

        # 1) Tokenize as a batch: [B, L]
        tokenized = tok.tokenize(texts).to(dev)  # COCO captions are short, padding is fine
        B, L = tokenized.shape

        # 2) Hook last-K blocks once, for the whole batch
        Hs = []

        def hook_fn(_mod, _inp, out):
            Hs.append(out if isinstance(out, torch.Tensor) else out[0])

        blocks = m.text_model.backbone.blocks
        K = min(self.last_k, len(blocks))
        handles = [b.register_forward_hook(hook_fn) for b in blocks[-K:]]

        _ = m.encode_text(tokenized)  # [B, 2D], but we ignore this output; we only want hidden states

        for h in handles:
            h.remove()

        assert len(Hs) == K, f"expected {K} hooked states, got {len(Hs)}"
        # Hs[k]: [B, L, hidden_dim]
        Hseq = torch.stack(Hs, 0).mean(0)       # [B, L, hidden_dim]
        Hseq = m.text_model.backbone.ln_final(Hseq)

        # 3) Project with same head as encode_text
        head = (
            m.text_model.text_model.head
            if hasattr(m.text_model, "text_model")
            else m.text_model.head
        )
        proj = head.linear_projection                   # [hidden_dim -> 2D]
        T2D = proj(Hseq)                                # [B, L, 2D]
        D2 = T2D.shape[-1]
        D = D2 // 2
        Tloc = T2D[..., D:]                             # [B, L, D] (patch-aligned half)

        # 4) L2 normalize per token
        Tloc = Tloc / (Tloc.norm(dim=-1, keepdim=True) + 1e-8)  # [B, L, D]

        return Tloc

    def itm_forward_from_tokens(
        self,
        img_tokens: torch.Tensor,   # [B, P, D]
        txt_tokens: torch.Tensor,   # [B, L, D]
    ) -> torch.Tensor:
        """
        ITM forward given precomputed tokens.
        Uses the fused CLS token as joint representation (ALBEF-style).
        Returns logits [B, 2].
        """
        self.model.eval()
        self.fusion.train()

        fused_txt, _ = self.fusion(txt_tokens, img_tokens, return_attn=False)  # [B, 1+L, D]

        cls_repr = fused_txt[:, 0, :]             # [B, D]  <-- CLS
        logits = self.fusion.itm_head(cls_repr)   # [B, 2]
        return logits

    # ---------- main API ----------

    def get_heatmaps(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """
        Grad-CAM heatmaps based on CLS→patch cross-attention in the LAST
        multimodal block, using the ITM 'match' logit as the target.

        images: [B_img, 3, H, W]
        texts:  list[str] of length B_txt

        returns:
            heatmaps: [B_txt, 1, H, W] in [0, 1]
        """
        device = self.device
        self.model.eval()
        self.fusion.eval()

        # ---- 1) Get frozen image/text tokens (no grad needed here) ----
        B_img_in, _, H, W = images.shape

        with torch.no_grad():
            img_tokens = self._get_visual_tokens(images)       # [B_img, P, D]
            txt_tokens = self._get_text_token_batch(texts)     # [B_txt, L, D]

        B_img, P, D = img_tokens.shape
        B_txt, L, D2 = txt_tokens.shape
        assert D == D2, f"D mismatch: img={D}, txt={D2}"

        # Expand single image for multiple texts
        if B_img == 1 and B_txt > 1:
            img_tokens = img_tokens.expand(B_txt, P, D)
            B_img = B_txt
        elif B_img != B_txt:
            raise RuntimeError(
                f"Batch mismatch: img_tokens={img_tokens.shape}, "
                f"txt_tokens={txt_tokens.shape}, len(texts)={len(texts)}"
            )

        # Detach to make them leaf tensors for this Grad-CAM graph
        img_tokens = img_tokens.to(device=device)
        txt_tokens = txt_tokens.to(device=device)

        img_tokens.requires_grad_(False)
        txt_tokens.requires_grad_(False)

        # ---- 2) Forward through fusion and capture last-layer cross-attn ----
        self.fusion.zero_grad(set_to_none=True)
        fused_txt, attn = self.fusion(txt_tokens, img_tokens, return_attn=True)
        # fused_txt: [B_txt, 1+L, D]
        # attn:      [B_txt, H, 1+L, P]

        if attn is None:
            raise RuntimeError("Fusion encoder did not return attention maps.")
        
        if not isinstance(attn, torch.Tensor):
            raise RuntimeError(f"Expected attn to be Tensor, got {type(attn)}")

        # We want gradients w.r.t. attn
        attn.retain_grad()

        # ---- 3) Compute ITM 'match' score from CLS and backprop ----
        cls_repr = fused_txt[:, 0, :]                # [B_txt, D]
        logits = self.fusion.itm_head(cls_repr)      # [B_txt, 2]

        # class 1 = "matched" (same as your ITM labels)
        score = logits[:, 1].sum()                   # scalar
        score.backward(retain_graph=False)

        grads = attn.grad                            # [B_txt, H, 1+L, P]

        if grads is None:
            print("[WARN] attn.grad is None, falling back to raw attention heatmaps.")

        # ---- 4) Grad-CAM on CLS → patch attention ----
        # CLS token is at index 0 along the (1+L) axis
        A_cls = attn[:, :, 0, :]                     # [B_txt, H, P]
        G_cls = grads[:, :, 0, :]                    # [B_txt, H, P]

        # Head weights α_h: average gradient over patches
        alpha = G_cls.mean(dim=-1, keepdim=True)     # [B_txt, H, 1]

        # Weighted sum over heads
        cam = (alpha * A_cls).sum(dim=1)             # [B_txt, P]

        # ReLU and per-sample normalization
        cam = F.relu(cam)
        max_vals = cam.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        cam = cam / max_vals                         # [B_txt, P] in [0,1]

        # ---- 5) Reshape to spatial map and upsample ----
        H_p = W_p = int(P ** 0.5)
        assert H_p * W_p == P, f"cannot reshape P={P} into square grid"

        heat = cam.view(B_txt, 1, H_p, W_p)          # [B_txt, 1, H_p, W_p]
        heat = F.interpolate(heat, size=(H, W), mode="bicubic", align_corners=False)

        # Final clamp to [0,1]
        heat = heat.clamp(0.0, 1.0)

        return heat
    

def load_dinotxt_fusion_backend(
    device: str = "cuda",
    fusion_ckpt: str | None = None,
    **kwargs,
) -> DinoTxtFusionBackend:
    """
    Factory used by inference_arpgrounding.py to build the DinoTxt fusion backend.
    """
    backend = DinoTxtFusionBackend(device=device, **kwargs)

    if fusion_ckpt is not None:
        state = torch.load(fusion_ckpt, map_location=device)
        fusion_state = state.get("fusion", state)
        backend.fusion.load_state_dict(fusion_state, strict=True)

    backend.model.eval()
    backend.fusion.eval()
    return backend