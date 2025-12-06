import numpy as np
import torch.utils.data
from tqdm import tqdm

from arp_backends.dinotxt import load_dinotxt_fusion_backend

from datasets.flicker import get_flicker1K_dataset
from datasets.referit import get_refit_test_dataset
from datasets.visual_genome import get_VGtest_dataset

try:
    import CLIP.clip as clip
except Exception:
    pass

try:
    from lavis.models import load_model_and_preprocess
except Exception:
    pass

try:
    from meter.config import config as meter_config
    from meter.datamodules.datamodule_base import get_pretrained_tokenizer
    from meter.modules import METERTransformerSS
    from meter.transforms import keys_to_transforms as meter_keys_to_transforms
except Exception:
    pass

from utils import interpret_albef, interpret_blip, interpret_clip, interpret_meter
from utils_grounding import calc_correctness, no_tuple


def inference_clip(ds, clip_model, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
    for i, (real_imgs, meta, size, _) in enumerate(pbar):
        if len(list(meta.keys())) == 0:
            continue
        real_imgs = real_imgs.cuda()
        size = [int(size[0]), int(size[1])]
        if args["dataset"] == "flicker" or args["task"] == "vg_train" or args["task"] == "coco":
            for sen in meta.keys():
                item = meta[sen]
                title, bbox = no_tuple(item["sentences"]), item["bbox"]
                text = clip.tokenize(title).to("cuda")
                curr_image = real_imgs.repeat(text.shape[0], 1, 1, 1)
                heatmap = interpret_clip(curr_image, text, clip_model, "cuda")
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
                bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        else:
            text = []
            bboxes = []
            for item in list(meta.values()):
                text.append(item["sentences"][0])
                bboxes.append(item["bbox"])
            text_tokens = clip.tokenize(text).to("cuda")
            curr_image = real_imgs.repeat(text_tokens.shape[0], 1, 1, 1)
            index = np.cumsum(np.ones(text_tokens.shape[0])).astype(np.uint8) - 1
            heatmaps = interpret_clip(curr_image, text_tokens, clip_model, "cuda")
            for k, heatmap in enumerate(heatmaps):
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy().astype(np.float)
                bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        bbox_correctness = 100.0 * cnt_correct / cnt_overall
        hit_correctness = 100.0 * cnt_correct_hit / cnt_overall
        att_correctness = 100.0 * att_correct / cnt_overall
        prnt = "bbox_correctness:{:.2f}; hit_correctness:{:.2f}; att_correctness:{:.2f}".format(
            bbox_correctness, hit_correctness, att_correctness
        )
        pbar.set_description(prnt)
    print(prnt)


def meter_make_batch(image, text, tokenizer, max_text_len):
    tokenized_text = tokenizer(
        text, 
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_special_tokens_mask=True,
    )
    batch = dict()
    batch["text_ids"] = torch.tensor(tokenized_text["input_ids"]).cuda()
    batch["text_labels"] = torch.tensor(tokenized_text["input_ids"]).cuda()
    batch["text_masks"] = torch.tensor(tokenized_text["attention_mask"]).cuda()
    curr_image = image.repeat(batch["text_ids"].shape[0], 1, 1, 1).cuda()
    batch["image"] = [curr_image]

    return batch


def inference_meter(ds, meter_model, tokenizer, max_text_len, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
    for p in meter_model.parameters():
        p.requires_grad = False
    for i, (real_imgs, meta, size, _) in enumerate(pbar):
        try:
            if len(list(meta.keys())) == 0:
                continue
            real_imgs = real_imgs.cuda()
            size = [int(size[0]), int(size[1])]
            if args["dataset"] == "flicker" or args["task"] == "vg_train" or args["task"] == "coco":
                for sen in meta.keys():
                    item = meta[sen]
                    title, bbox = no_tuple(item["sentences"]), item["bbox"]
                    batch = meter_make_batch(real_imgs, title, tokenizer, max_text_len)
                    heatmap = interpret_meter(batch, meter_model, "cuda")
                    heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
                    bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
                    cnt_correct += bbox_c
                    cnt_correct_hit += hit_c
                    att_correct += att_c
                    cnt_overall += 1
            else:
                text = []
                bboxes = []
                for item in list(meta.values()):
                    text.append(item["sentences"][0])
                    bboxes.append(item["bbox"])
                text_tokens = tokenizer(
                    text, 
                    padding="max_length",
                    truncation=True,
                    max_length=max_text_len,
                    return_special_tokens_mask=True,
                )
                batch = dict()
                batch["text_ids"] = torch.tensor(text_tokens["input_ids"]).cuda()
                batch["text_labels"] = torch.tensor(text_tokens["input_ids"]).cuda()
                batch["text_masks"] = torch.tensor(text_tokens["attention_mask"]).cuda()
                curr_image = real_imgs.repeat(batch["text_ids"].shape[0], 1, 1, 1)
                batch["image"] = [curr_image]
                heatmaps = interpret_meter(batch, meter_model, "cuda")
                for k, heatmap in enumerate(heatmaps):
                    heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy().astype(np.float)
                    bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap.astype(np.float), size)
                    cnt_correct += bbox_c
                    cnt_correct_hit += hit_c
                    att_correct += att_c
                    cnt_overall += 1
            bbox_correctness = 100.0 * cnt_correct / cnt_overall
            hit_correctness = 100.0 * cnt_correct_hit / cnt_overall
            att_correctness = 100.0 * att_correct / cnt_overall
            prnt = "bbox_correctness:{:.2f}; hit_correctness:{:.2f}; att_correctness:{:.2f}".format(
                bbox_correctness, hit_correctness, att_correctness
            )
            pbar.set_description(prnt)
        except Exception as e:  # skip OOM samples
            print(str(e))
    print(prnt)


def inference_albef(ds, model, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
    for i, (real_imgs, meta, size, _) in enumerate(pbar):
        try:
            if len(list(meta.keys())) == 0:
                continue
            real_imgs = real_imgs.cuda()
            size = [int(size[0]), int(size[1])]
            if args["dataset"] == "flicker" or args["task"] == "vg_train" or args["task"] == "coco":
                for sen in meta.keys():
                    item = meta[sen]
                    title, bbox = no_tuple(item["sentences"]), item["bbox"]
                    curr_image = real_imgs.repeat(len(title), 1, 1, 1)
                    heatmap = interpret_albef(curr_image, title, model, "cuda")
                    heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
                    bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
                    cnt_correct += bbox_c
                    cnt_correct_hit += hit_c
                    att_correct += att_c
                    cnt_overall += 1
            else:
                text = []
                bboxes = []
                for item in list(meta.values()):
                    text.append(item["sentences"][0])
                    bboxes.append(item["bbox"])
                curr_image = real_imgs.repeat(len(text), 1, 1, 1)
                heatmaps = interpret_albef(curr_image, text, model, "cuda")
                for k, heatmap in enumerate(heatmaps):
                    heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy().astype(np.float)
                    bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap.astype(np.float), size)
                    cnt_correct += bbox_c
                    cnt_correct_hit += hit_c
                    att_correct += att_c
                    cnt_overall += 1
            bbox_correctness = 100.0 * cnt_correct / cnt_overall
            hit_correctness = 100.0 * cnt_correct_hit / cnt_overall
            att_correctness = 100.0 * att_correct / cnt_overall
            prnt = "bbox_correctness:{:.2f}; hit_correctness:{:.2f}; att_correctness:{:.2f}".format(
                bbox_correctness, hit_correctness, att_correctness
            )
            pbar.set_description(prnt)
        except Exception as e:  # skip OOM samples
            print(str(e))
    print(prnt)


def inference_blip(ds, blip_model, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0
    for i, (real_imgs, meta, size, _) in enumerate(pbar):
        if len(list(meta.keys())) == 0:
            continue
        real_imgs = real_imgs.cuda()
        size = [int(size[0]), int(size[1])]
        if args["dataset"] == "flicker" or args["task"] == "vg_train" or args["task"] == "coco":
            for sen in meta.keys():
                item = meta[sen]
                title, bbox = no_tuple(item["sentences"]), item["bbox"]
                curr_image = real_imgs.repeat(len(title), 1, 1, 1)
                heatmap = interpret_blip(curr_image, title, blip_model, "cuda")
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy()
                bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        else:
            text = []
            bboxes = []
            for item in list(meta.values()):
                text.append(item["sentences"][0])
                bboxes.append(item["bbox"])
            curr_image = real_imgs.repeat(len(text), 1, 1, 1)
            heatmaps = interpret_blip(curr_image, text, blip_model, "cuda")
            for k, heatmap in enumerate(heatmaps):
                heatmap = heatmap.mean(dim=0).squeeze().detach().clone().cpu().numpy().astype(np.float)
                bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap.astype(np.float), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1
        bbox_correctness = 100.0 * cnt_correct / cnt_overall
        hit_correctness = 100.0 * cnt_correct_hit / cnt_overall
        att_correctness = 100.0 * att_correct / cnt_overall
        prnt = "bbox_correctness:{:.2f}; hit_correctness:{:.2f}; att_correctness:{:.2f}".format(
            bbox_correctness, hit_correctness, att_correctness
        )
        pbar.set_description(prnt)
    print(prnt)

def inference_dinotxt(ds, dinotxt_backend, args):
    pbar = tqdm(ds)
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    att_correct = 0

    for i, (real_imgs, meta, size, _) in enumerate(pbar):
        if len(list(meta.keys())) == 0:
            continue
        real_imgs = real_imgs.cuda()
        size = [int(size[0]), int(size[1])]

        if args["dataset"] == "flicker":
            # same as before, but we can also chunk if needed
            for sen in meta.keys():
                item = meta[sen]
                titles, bbox = no_tuple(item["sentences"]), item["bbox"]
                curr_image = real_imgs.repeat(len(titles), 1, 1, 1)
                heatmaps = dinotxt_backend.get_heatmaps(curr_image, titles)  # [T, 1, H, W]
                heatmap = heatmaps.mean(dim=0).squeeze(0).detach().clone().cpu().numpy()
                bbox_c, hit_c, att_c = calc_correctness(bbox, heatmap.astype(np.float32), size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1

        else:
            # ------- CHUNKED VERSION FOR VG / REFIT -------
            texts = []
            bboxes = []
            for item in list(meta.values()):
                texts.append(item["sentences"][0])
                bboxes.append(item["bbox"])

            all_heatmaps = []
            base_image = real_imgs  # [1, 3, H, W]

            chunk_size = 4  # you can try 8 if memory allows

            for start in range(0, len(texts), chunk_size):
                end = start + chunk_size
                chunk_texts = texts[start:end]
                # repeat image only for this small chunk
                curr_image = base_image.repeat(len(chunk_texts), 1, 1, 1)
                hm_chunk = dinotxt_backend.get_heatmaps(curr_image, chunk_texts)  # [C, 1, H, W]
                all_heatmaps.append(hm_chunk.cpu())

            heatmaps = torch.cat(all_heatmaps, dim=0)  # [N, 1, H, W], N == len(texts)

            for k, heatmap in enumerate(heatmaps):
                heatmap = heatmap.mean(dim=0).squeeze().numpy().astype(np.float32)
                bbox_c, hit_c, att_c = calc_correctness(bboxes[k], heatmap, size)
                cnt_correct += bbox_c
                cnt_correct_hit += hit_c
                att_correct += att_c
                cnt_overall += 1

        bbox_correctness = 100.0 * cnt_correct / cnt_overall
        hit_correctness = 100.0 * cnt_correct_hit / cnt_overall
        att_correctness = 100.0 * att_correct / cnt_overall
        prnt = "bbox_correctness:{:.2f}; hit_correctness:{:.2f}".format(
            bbox_correctness, hit_correctness, att_correctness
        )
        pbar.set_description(prnt)

    print(prnt)


def main(args=None):
    gpu_num = torch.cuda.device_count()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args["dataset"] == "flicker":
        testset = get_flicker1K_dataset(args=args)
    elif args["dataset"] == "refit":
        testset = get_refit_test_dataset(args=args)
    elif args["dataset"] == "vg":
        testset = get_VGtest_dataset(args=args)
    ds = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=int(args["nW"]), shuffle=False, drop_last=False)

    if args["clip_eval"]:
        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        inference_clip(ds, clip_model, args)
    elif args["meter_eval"]:
        config = meter_config()
        meter_model = METERTransformerSS(config).to(device)
        transform = meter_keys_to_transforms(config["val_transform_keys"], size=config["image_size"])[0]
        testset.transform = transform
        meter_tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        inference_meter(ds, meter_model, meter_tokenizer, config["max_text_len"], args)
    elif args["albef_eval"]:
        albef_model, _, _ = load_model_and_preprocess("albef_image_text_matching", "coco", device=device, is_eval=True)
        if args["albef_path"]:
            checkpoint = torch.load(args["albef_path"], map_location="cpu")
            state_dict = checkpoint["state_dict"]
            albef_model.load_state_dict(state_dict)
        inference_albef(ds, albef_model, args)
    elif args["blip2_eval"]:
        blip_model, _, _ = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
        if args["blip2_path"]:
            checkpoint = torch.load(args["blip2_path"], map_location="cpu")
            state_dict = checkpoint["state_dict"]
            blip_model.load_state_dict(state_dict)
        inference_blip(ds, blip_model, args)
    elif args["dinotxt_eval"]:
        dinotxt_backend = load_dinotxt_backend(device=device)
        testset.transform = dinotxt_backend.preprocess
        ds = torch.utils.data.DataLoader(
            testset, batch_size=1, num_workers=int(args["nW"]), shuffle=False, drop_last=False
        )
        inference_dinotxt(ds, dinotxt_backend, args)


def resize_then_center_crop_bbox(bbox, orig_size, out_size):
    """
    bbox: [x1, y1, x2, y2] in original image coords
    orig_size: (H0, W0)   (from `size`)
    out_size: int (Isize) final square side after Resize+CenterCrop
    """
    H0, W0 = orig_size
    s = float(out_size)
    x1, y1, x2, y2 = map(float, bbox)

    # Resize with aspect ratio preserved (short side -> s)
    if H0 <= W0:
        # height is short side
        scale = s / H0
        H1 = s
        W1 = W0 * scale
        crop_x0 = (W1 - s) / 2.0  # horizontal crop offset
        crop_y0 = 0.0
    else:
        # width is short side
        scale = s / W0
        W1 = s
        H1 = H0 * scale
        crop_x0 = 0.0
        crop_y0 = (H1 - s) / 2.0  # vertical crop offset

    # Resize bbox
    x1_r = x1 * scale
    x2_r = x2 * scale
    y1_r = y1 * scale
    y2_r = y2 * scale

    # Center crop bbox
    x1_c = x1_r - crop_x0
    x2_c = x2_r - crop_x0
    y1_c = y1_r - crop_y0
    y2_c = y2_r - crop_y0

    # Clamp into [0, s]
    x1_c = max(0.0, min(s, x1_c))
    x2_c = max(0.0, min(s, x2_c))
    y1_c = max(0.0, min(s, y1_c))
    y2_c = max(0.0, min(s, y2_c))

    return [
        int(round(x1_c)),
        int(round(y1_c)),
        int(round(x2_c)),
        int(round(y2_c)),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-nW", "--nW", default=0, help="number of workers", required=False)
    parser.add_argument("-Isize", "--Isize", default=304, help="image size", required=False)
    parser.add_argument("-clip_eval", "--clip_eval", default=False, action="store_true", help="eval clip", required=False)
    parser.add_argument("-meter_eval", "--meter_eval", default=False, action="store_true", help="eval meter", required=False)
    parser.add_argument("-albef_eval", "--albef_eval", default=False, action="store_true", help="eval albef", required=False)
    parser.add_argument("-albef_path", "--albef_path", type=str, default="", help="albef folder path", required=False)
    parser.add_argument("-blip_eval", "--blip_eval", default=False, action="store_true", help="eval blip", required=False)
    parser.add_argument("-blip2_eval", "--blip2_eval", default=False, action="store_true", help="eval blip2", required=False)
    parser.add_argument("-blip2_path", "--blip2_path", type=str, default="", help="blip2 folder path", required=False)
    parser.add_argument("-data_path", "--data_path", default="/path_to_data/cars", help="data set path", required=False)
    parser.add_argument("-val_path", "--val_path", default="", help="data set path", required=False)
    parser.add_argument("-dataset", "--dataset", default="flicker", help="dataset task", required=False)
    parser.add_argument("-img_path", "--img_path", default=1, help="dataset task", required=False)
    parser.add_argument(
        "-dinotxt_eval", "--dinotxt_eval",
        default=False, action="store_true",
        help="use dino.txt backend", required=False
    )
    args = vars(parser.parse_args())

    main(args=args)
