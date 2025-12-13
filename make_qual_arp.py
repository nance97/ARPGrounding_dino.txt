import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from datasets.arpgrounding import get_dataset
from inference_grounding import interpret_clip
import CLIP.clip as clip

from arp_backends.dinotxt import load_dinotxt_backend

def tensor_to_vis(img_3hw: torch.Tensor):
    # img_3hw: float tensor, likely normalized; we just scale for display.
    # This is "good enough" for overlays. If you have mean/std, you can unnormalize.
    x = img_3hw.detach().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    x = x.permute(1, 2, 0).numpy()  # HWC
    return x

def draw(ax, img_vis, hm, bbox, title):
    ax.imshow(img_vis)
    ax.imshow(hm, alpha=0.45)
    x1, y1, x2, y2 = bbox
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=2))
    ax.set_title(title, fontsize=8)
    ax.axis("off")

@torch.no_grad()
def normalize01(hm: torch.Tensor):
    hm = torch.relu(hm)
    m = hm.max().clamp(min=1e-6)
    return (hm / m)

def extract_one(uid, split, out_path, val_path, device="cuda"):
    global_idx, local_idx = uid

    # ---------- dataset (same as their script) ----------
    args = {
        "nW": 0,
        "Isize": 304,   # doesn't matter much if we resize ourselves; keep consistent with their setup
        "val_path": val_path,
        "split": split,
    }
    ds = get_dataset(args)
    ds.files = list(ds.annotations.keys())
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    # ---------- models ----------
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

    dinotxt_backend = load_dinotxt_backend(device=device)
    # IMPORTANT: do NOT set ds.transform = dinotxt_backend.preprocess here,
    # because we want to reuse the same fetched sample; we will resize the tensor ourselves.

    # ---------- find the sample ----------
    target = None
    for i, inputs in enumerate(dl):
        real_imgs, meta, size, _ = inputs
        if len(list(meta.keys())) == 0:
            continue
        if i == global_idx:
            target = (real_imgs, meta, size)
            break

    if target is None:
        raise RuntimeError(f"Could not find global_idx={global_idx} (was it skipped due to empty meta?)")

    real_imgs, meta, size = target
    real_imgs = real_imgs.to(device)   # [1,3,H,W]
    size = [int(size[0]), int(size[1])]  # original H0,W0

    # meta is a dict; items_list corresponds to list(meta.values()) in their script
    items_list = list(meta.values())
    if local_idx >= len(items_list):
        raise RuntimeError(f"local_idx={local_idx} out of range; sample only has {len(items_list)} items")

    item = items_list[local_idx]
    pos_text = item["sentences"][0]
    neg_text = item["neg_sentences"][0]

    # ---------- choose a common render/eval resolution ----------
    # CLIP expects 224x224 in their eval; let's standardize to 224 for BOTH backends for clean visuals.
    Ht = Wt = 224
    img_224 = F.interpolate(real_imgs, size=(Ht, Wt), mode="bilinear", align_corners=False)

    # ---------- scale bboxes to 224x224 exactly like their scaling logic ----------
    # item["bbox"][0/1] are in original image coordinates (H0,W0). Convert to 224 scale.
    def scale_bbox(b):
        x1 = int(b[0] / size[1] * Wt)
        y1 = int(b[1] / size[0] * Ht)
        x2 = int(b[2] / size[1] * Wt)
        y2 = int(b[3] / size[0] * Ht)
        return [x1, y1, x2, y2]

    pos_bbox = scale_bbox(item["bbox"][0])
    neg_bbox = scale_bbox(item["bbox"][1])

    # ---------- CLIP heatmaps ----------
    tok_pos = clip.tokenize([pos_text]).to(device)
    tok_neg = clip.tokenize([neg_text]).to(device)

    # interpret_clip expects [B,3,224,224]
    hm_clip_pos = interpret_clip(img_224, tok_pos, clip_model, device=device)[0, 0]
    hm_clip_neg = interpret_clip(img_224, tok_neg, clip_model, device=device)[0, 0]
    hm_clip_pos = normalize01(hm_clip_pos).detach().cpu().numpy()
    hm_clip_neg = normalize01(hm_clip_neg).detach().cpu().numpy()

    # ---------- dino.txt heatmaps ----------
    # Your backend expects images already; it will upsample internally to input H,W (224 here).
    hm_dino_pos = dinotxt_backend.get_heatmaps(img_224, [pos_text])[0, 0]
    hm_dino_neg = dinotxt_backend.get_heatmaps(img_224, [neg_text])[0, 0]
    hm_dino_pos = normalize01(hm_dino_pos).detach().cpu().numpy()
    hm_dino_neg = normalize01(hm_dino_neg).detach().cpu().numpy()

    # ---------- visualize ----------
    img_vis = tensor_to_vis(img_224[0])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    draw(axes[0,0], img_vis, hm_clip_pos, pos_bbox, f"CLIP – {pos_text}")
    draw(axes[0,1], img_vis, hm_clip_neg, neg_bbox, f"CLIP – {neg_text}")
    draw(axes[1,0], img_vis, hm_dino_pos, pos_bbox, f"dino.txt – {pos_text}")
    draw(axes[1,1], img_vis, hm_dino_neg, neg_bbox, f"dino.txt – {neg_text}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

if __name__ == "__main__":
    # Example:
    # uid = (global_idx, local_idx) from your paired dataframe for priority
    uid = (6, 0)
    split = "priority"
    extract_one(uid, split, out_path="img/ARPG_priority_qual/qual_example.pdf", device="cuda" if torch.cuda.is_available() else "cpu")
