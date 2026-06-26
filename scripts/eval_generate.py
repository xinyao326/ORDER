import argparse
import os
import torch
import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.model_config import config_dict
from skimage.metrics import peak_signal_noise_ratio as psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default='composite', choices=['composite','fiber'])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--backbone", type=str, default='CLIP_ViT-B/16')
    parser.add_argument("--config", type=str, default='order')
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--prefix", type=str, default='order_dyn',
                        choices=['order_dyn', 'order_alpha', 'matmcl_imagenet', 'matmcl_clip'])
    parser.add_argument("--n_threads", type=int, default=2)
    parser.add_argument("--split", type=str, default='train', choices=['train','test','combined'])
    parser.add_argument("--split_suffix", type=str, default='_clean',
                        help="CSV suffix: '_clean' (default) or '_surr' for surrogate variants.")
    parser.add_argument("--label_ratio", type=float, default=1.0,
                        help="Must match the label_ratio used during pretraining (1.0, 0.5, or 0.2).")
    parser.add_argument("--matmcl_ckpt", type=str, default=None,
                        help="Ignored; accepted for compatibility with run_generation.sh COMMON_ARGS.")
    args = parser.parse_args()

    ss = args.split_suffix
    ratio_tag = '' if args.label_ratio == 1.0 else f'_semi{args.label_ratio}'
    if args.prefix == 'order_alpha':
        setting = f'seed{args.seed}_Alpha{str(args.alpha)}{ss}{ratio_tag}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix, args.backbone.replace('/','_'), setting)
    elif args.prefix == 'order_dyn':
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix, args.backbone.replace('/','_'), setting)
    else:  # matmcl_imagenet / matmcl_clip
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix, setting)
    if not os.path.exists(args.savepth):
        raise RuntimeError(f'path {args.savepth} not exists')
    args.logger = create_logger(args.savepth, f'{args.log}-{args.split}-EvalGen')
    args.weightpth = os.path.join(args.savepth, "weight-final.pth")
    args.setting = setting
    args.logger.info(print_arg(args))

    return args


def null_sync(t, *args, **kwargs):
    return [t]


def get_image(path, grey=False):
    if not grey:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    ori_img = Image.open(path).convert('RGB')
    img = transform(ori_img)
    return img


def main(args):
    metrics = {}
    FID = {'feature': 2048}
    IS = {'feature': 'logits_unbiased'}
    KID = {'feature': 2048}
    LPIPS = {'net_type': 'alex'}
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.split == 'combined':
        splits = ['train', 'test']
    else:
        splits = [args.split]
    filelis = []
    for sp in splits:
        d = os.path.join(args.savepth, f'gen-{sp}')
        filelis.extend(os.path.join(d, idx) for idx in os.listdir(d))
    all_real, all_gen, all_gen_group = [], [], []
    for idx in filelis:
        cur_path = idx
        realimg_pth = os.path.join(cur_path, 'real.png')
        real_img = get_image(realimg_pth)
        gen_list = os.listdir(cur_path)
        cur_gen = []
        for gen in gen_list:
            if gen != 'real.png':
                cur_gen_path = os.path.join(cur_path, gen)
                all_gen.append(get_image(cur_gen_path, grey=True if args.dataset == 'composite' else False))
                cur_gen.append(get_image(cur_gen_path, grey=True if args.dataset == 'composite' else False))
        if len(all_gen_group) == 0 or len(cur_gen) == all_gen_group[-1].shape[0]:
            all_real.append(real_img)
            all_gen_group.append(torch.stack(cur_gen))  # [9, 3,224,224]

    # Keep all image tensors on CPU; move to device only in small batches to save GPU memory.
    all_gen_group = torch.stack(all_gen_group, 0)  # [N, K, 3, 224, 224] — CPU
    all_real = torch.stack(all_real, 0)             # [N, 3, 224, 224]    — CPU
    all_gen = torch.stack(all_gen, 0)               # [N*K, 3, 224, 224]  — CPU

    int_real_images = all_real.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)      # CPU
    int_generated_images = all_gen.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)  # CPU
    gen_num = int_generated_images.shape[0]
    btz = 256

    fid = FrechetInceptionDistance(**FID, dist_sync_fn=null_sync).to(args.device)
    inception = InceptionScore(**IS, dist_sync_fn=null_sync).to(args.device)
    kernel_inception = KernelInceptionDistance(**KID, dist_sync_fn=null_sync, subset_size=int(len(all_real)/2)).to(args.device)

    # Feed real images in batches
    for start in range(0, int_real_images.shape[0], btz):
        batch = int_real_images[start:start+btz].to(args.device)
        fid.update(batch, real=True)
        kernel_inception.update(batch, real=True)

    # Feed generated images in batches
    start, end = 0, btz
    while True:
        cur_gen_images = int_generated_images[start:end].to(args.device)
        fid.update(cur_gen_images, real=False)
        kernel_inception.update(cur_gen_images, real=False)
        inception.update(cur_gen_images)
        if end >= gen_num:
            break
        start = end
        end = min(end + btz, gen_num)

    metrics["FID"] = fid.compute().item()

    is_mean, is_std = inception.compute()
    metrics["IS_mean"] = is_mean.item()
    metrics["IS_std"] = is_std.item()

    kid_mean, kid_std = kernel_inception.compute()
    metrics["KID_mean"] = kid_mean.item()
    metrics["KID_std"] = kid_std.item()

    metrics["LPIPS"] = 0
    metrics["PSNR"] = 0
    n_gen = all_gen_group.shape[1]
    for i in range(n_gen):
        renorm_real = all_real.mul(2).sub(1).clamp(-1, 1).to(args.device)
        renorm_gen = all_gen_group[:, i].mul(2).sub(1).clamp(-1, 1).to(args.device)
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS, dist_sync_fn=null_sync).to(args.device)
        lpips.update(renorm_real, renorm_gen)
        metrics["LPIPS"] += lpips.compute().item() / n_gen
        metrics["PSNR"] += psnr(renorm_real.cpu().numpy(), renorm_gen.cpu().numpy()) / n_gen
        del lpips, renorm_real, renorm_gen
        torch.cuda.empty_cache()

    args.logger.info(f'{int_real_images.shape[0]} real, {int_generated_images.shape[0]} generated samples')
    args.logger.info(metrics)


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
