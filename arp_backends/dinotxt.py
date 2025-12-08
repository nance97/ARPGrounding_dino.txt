import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.hub.dinotxt import (
    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
    get_tokenizer,
)
from dinov2.data.transforms import make_classification_eval_transform
import math


class DifferentiableMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # [B, L_q, D]
        key: torch.Tensor,     # [B, L_k, D]
        value: torch.Tensor,   # [B, L_k, D]
        need_weights: bool = False,
    ):
        B, L_q, D = query.shape
        Bk, L_k, Dk = key.shape
        assert B == Bk and D == Dk == self.d_model

        # Linear projections
        Q = self.q_proj(query)   # [B, L_q, D]
        K = self.k_proj(key)     # [B, L_k, D]
        V = self.v_proj(value)   # [B, L_k, D]

        # Reshape to [B, H, L, head_dim]
        H = self.num_heads
        Hd = self.head_dim

        Q = Q.view(B, L_q, H, Hd).transpose(1, 2)  # [B, H, L_q, Hd]
        K = K.view(B, L_k, H, Hd).transpose(1, 2)  # [B, H, L_k, Hd]
        V = V.view(B, L_k, H, Hd).transpose(1, 2)  # [B, H, L_k, Hd]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Hd)   # [B, H, L_q, L_k]
        attn = F.softmax(scores, dim=-1)                               # [B, H, L_q, L_k]
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)                                # [B, H, L_q, Hd]

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, L_q, D) # [B, L_q, D]
        out = self.out_proj(context)
        out = self.proj_dropout(out)

        if need_weights:
            # Return full attention (L_q x L_k) per head, in graph
            return out, attn
        else:
            return out, None
        
class MultiModalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if dim_ff is None:
            dim_ff = 4 * d_model

        # Self-attention over text tokens
        self.self_attn = DifferentiableMultiHeadAttention(
            d_model=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )

        # Cross-attention: text queries, image keys/values
        self.cross_attn = DifferentiableMultiHeadAttention(
            d_model=d_model,
            num_heads=n_heads,
            dropout=dropout,
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
        img_tokens: [B, P, D]
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
            need_weights=need_attn,  # attn will now be in the graph
        )
        x = x + self.dropout(ca_out)
        x = self.norm2(x)

        # 3) feed-forward
        ff_out = self.ffn(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return (x, attn) if need_attn else (x, None)


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiModalBlock(d_model, n_heads) for _ in range(num_layers)]
        )
        # ITM head
        self.itm_head = nn.Linear(d_model, 2)

        # --- NEW: MLM head and [MASK] embedding ---
        self.mlm_head = nn.Linear(d_model, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        with torch.no_grad():
            self.mask_token[:] = self.mask_token / (self.mask_token.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, txt_tokens, img_tokens, return_attn: bool = False):
        """
        txt_tokens: [B, L, D]
        img_tokens: [B, P, D]
        """
        x = txt_tokens  # no extra CLS
        
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

    def get_heatmaps(self, images: torch.Tensor, texts: list[str], token_strategy: str = "cls") -> torch.Tensor:
        device = self.device
        self.model.eval()
        self.fusion.eval()

        B_img_in, _, H, W = images.shape

        with torch.no_grad():
            img_tokens = self._get_visual_tokens(images)
            txt_tokens = self._get_text_token_batch(texts)

        img_tokens = img_tokens.to(device)
        txt_tokens = txt_tokens.to(device)

        B_img, P, D = img_tokens.shape
        B_txt, L, D2 = txt_tokens.shape
        assert D == D2

        if B_img == 1 and B_txt > 1:
            img_tokens = img_tokens.expand(B_txt, P, D)
            B_img = B_txt
        elif B_img != B_txt:
            raise RuntimeError("batch mismatch")

        # enable grad on tokens? not strictly needed for attn-grad, but okay:
        img_tokens = img_tokens.detach()
        txt_tokens = txt_tokens.detach()

        self.fusion.zero_grad(set_to_none=True)

        fused_txt, attn = self.fusion(txt_tokens, img_tokens, return_attn=True)
        # attn: [B, H_heads, L, P]
        if attn is None:
            raise RuntimeError("no attention returned")

        attn.retain_grad()

        cls_repr = fused_txt[:, 0, :]
        logits = self.fusion.itm_head(cls_repr)

        score = (logits[:, 1] - logits[:, 0]).sum()
        score.backward()

        grads = attn.grad  # [B, H, L, P]
        if grads is None:
            raise RuntimeError("attn.grad is None")

        # weights: avg grad over patches
        weights = grads.mean(dim=-1, keepdim=True)   # [B, H, L, 1]
        cam = (weights * attn).sum(dim=1)            # [B, L, P]

        if token_strategy == "cls":
            cam_token = cam[:, 0, :]
        elif token_strategy == "max":
            norms = cam.norm(dim=-1)
            idx = norms.argmax(dim=-1)               # [B]
            cam_token = cam[torch.arange(B_txt, device=device), idx, :]
        else:
            raise ValueError(f"Unknown token_strategy={token_strategy}")

        cam_token = F.relu(cam_token)
        max_vals = cam_token.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        cam_token = cam_token / max_vals             # [B, P]

        H_p = W_p = int(P ** 0.5)
        heat = cam_token.view(B_txt, 1, H_p, W_p)
        heat = F.interpolate(heat, size=(H, W), mode="bicubic", align_corners=False)
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