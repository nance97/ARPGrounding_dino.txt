import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.hub.dinotxt import (
    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
    get_tokenizer,
)
from dinov2.data.transforms import make_classification_eval_transform


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
            query=x, key=img_tokens, value=img_tokens, need_weights=need_attn
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
        self.mlm_head = nn.Linear(d_model, d_model)  # predict original token features
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, txt_tokens, img_tokens, return_attn: bool = False):
        """
        txt_tokens: [B, L, D]
        img_tokens: [B, P, D]
        """
        x = txt_tokens
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
        ITM forward given precomputed tokens (no string → token conversion).
        Returns logits [B,2].
        """
        self.model.eval()
        self.fusion.train()

        fused_txt, _ = self.fusion(txt_tokens, img_tokens, return_attn=False)  # [B, L, D]
        pooled = fused_txt.mean(dim=1)  # [B, D]
        logits = self.fusion.itm_head(pooled)  # [B, 2]
        return logits

    # ---------- main API ----------

    @torch.no_grad()
    def get_heatmaps(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """
        images: [B_img, 3, H, W]  (ARPG: B_img == 1)
        texts:  list[str] of length B_txt (can be > 1 for multi-composition images)

        returns:
            heatmaps: [B_txt, 1, H, W]  (one heatmap per text)
        """
        # 1) Shapes
        B_img_in, _, H, W = images.shape

        # 2) Frozen tokens
        img_tokens = self._get_visual_tokens(images)     # [B_img, P, D]
        txt_tokens = self._get_text_token_batch(texts)   # [B_txt, L, D]

        B_img, P, D  = img_tokens.shape
        B_txt, L, D2 = txt_tokens.shape
        assert D == D2, f"D mismatch: img_tokens={D}, txt_tokens={D2}"

        # Debug (optional)
        # print(f"[DEBUG] B_img={B_img}, B_txt={B_txt}, P={P}, L={L}, len(texts)={len(texts)}")

        # 3) Make batch sizes match: replicate image for each text
        if B_img == 1 and B_txt > 1:
            # expand: [1, P, D] -> [B_txt, P, D] (no new memory, just view)
            img_tokens = img_tokens.expand(B_txt, P, D)
            B_img = B_txt
        elif B_img != B_txt:
            raise RuntimeError(
                f"Batch mismatch: img_tokens={img_tokens.shape}, "
                f"txt_tokens={txt_tokens.shape}, len(texts)={len(texts)}"
            )

        # 4) Run fusion encoder, ask for attention if you still want attn-based maps
        fused_txt, attn = self.fusion(txt_tokens, img_tokens, return_attn=True)
        # attn: [B_txt, n_heads, L, P]

        if attn is None:
            raise RuntimeError("Fusion encoder did not return attention maps.")

        B, n_heads, L_attn, P_attn = attn.shape
        assert B == B_txt and P_attn == P

        # 5) Choose which token to use (here: last token, like you already did)
        token_idx = L_attn - 1
        attn_token = attn[:, :, token_idx, :]      # [B_txt, n_heads, P]
        attn_mean  = attn_token.mean(dim=1)        # [B_txt, P]

        # 6) reshape to spatial map
        H_p = W_p = int(P ** 0.5)
        assert H_p * W_p == P, f"cannot reshape P={P} into square grid"
        heat = attn_mean.view(B_txt, 1, H_p, W_p)  # [B_txt, 1, H_p, W_p]

        # 7) upsample to image resolution
        heat = F.interpolate(heat, size=(H, W), mode="bicubic", align_corners=False)

        # 8) normalize to [0,1]
        heat = torch.relu(heat)
        flat = heat.view(B_txt, -1)
        max_vals = flat.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        heat = heat / max_vals.view(B_txt, 1, 1, 1)

        return heat  # [B_txt, 1, H, W]
    

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