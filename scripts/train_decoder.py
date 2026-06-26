import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.models import vit_b_16, ViT_B_16_Weights
from dalle2_pytorch import Unet, Decoder, DecoderTrainer
from torchvision.utils import save_image
from tqdm import tqdm

import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.models.order import OrderModel
from src.data.gen_dataset import ImageDataset
from src.model_config import config_dict


class _SGPTViT(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Linear(768, out_dim)

    def forward(self, _, x_img):
        return self.vit(x_img)


class _Encoder(nn.Module):
    def __init__(self, common_dim=128, latent_dim=128):
        super().__init__()
        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)


class SGPTCompatImageNet(nn.Module):
    def __init__(self, cond_dim, cardinality):
        super().__init__()
        from src.models import TableTransformerWrapper
        self.table_processor  = TableTransformerWrapper(
            in_dim=cond_dim, out_dim=128, dropout=0, cardinality=cardinality)
        self.vision_processor = _SGPTViT(out_dim=128)
        self.encoder          = _Encoder(128, 128)

    def encode(self, x, modal):
        if modal == 'tab':
            return self.encoder(self.table_processor(x, x))
        elif modal == 'image':
            return self.encoder(self.vision_processor(x, x))
        raise ValueError(f"Unknown modal: {modal}")


class SGPTCompatCLIP(nn.Module):
    def __init__(self, cond_dim, cardinality):
        super().__init__()
        from src.models import TableTransformerWrapper
        from src.models.myclip import peftCLIP
        self.table_processor  = TableTransformerWrapper(
            in_dim=cond_dim, out_dim=128, dropout=0, cardinality=cardinality)
        self.vision_processor = peftCLIP(
            model_name='openai/clip-vit-base-patch16', output_dim=128, lora_r=8)
        self.encoder          = _Encoder(128, 128)

    def encode(self, x, modal):
        if modal == 'tab':
            return self.encoder(self.table_processor(x, x))
        elif modal == 'image':
            return self.encoder(self.vision_processor(x, x))
        raise ValueError(f"Unknown modal: {modal}")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--n_epochs",      type=int,   default=None,
                        help="Training epochs. Default: 2000 for composite, 30 for fiber.")
    parser.add_argument("--log",           type=str,   default='log')
    parser.add_argument("--device",        type=str,   default="cuda:0")
    parser.add_argument("--dataset",       type=str,   default='composite',
                        choices=['composite', 'fiber'])
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--backbone",      type=str,   default='CLIP_ViT-B/16')
    parser.add_argument("--config",        type=str,   default='order')
    parser.add_argument("--alpha",         type=float, default=0.2)
    parser.add_argument("--r",             type=int,   default=8)
    parser.add_argument("--prefix",        type=str,   default='order_dyn',
                        choices=['order_dyn', 'order_alpha', 'matmcl_imagenet', 'matmcl_clip'])
    parser.add_argument("--n_threads",     type=int,   default=2)
    parser.add_argument("--split_suffix",  type=str,   default='_clean',
                        help="CSV suffix: '_clean' (default) or '_surr' for surrogate variants.")
    parser.add_argument("--matmcl_ckpt",   type=str,   default=None,
                        help="Path to MatMCL pretrained checkpoint (required for matmcl_* prefixes).")
    args = parser.parse_args()

    if args.n_epochs is None:
        args.n_epochs = 2000 if args.dataset == 'composite' else 30

    ss = args.split_suffix
    if args.prefix == 'order_dyn':
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', 'order_dyn',
                                    args.backbone.replace('/', '_'), setting)
    elif args.prefix == 'order_alpha':
        setting = f'seed{args.seed}_Alpha{args.alpha}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', 'order_alpha',
                                    args.backbone.replace('/', '_'), setting)
    else:  # matmcl_imagenet / matmcl_clip
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix, setting)
        os.makedirs(args.savepth, exist_ok=True)

    if args.prefix in ('order_dyn', 'order_alpha'):
        if not os.path.exists(args.savepth):
            raise RuntimeError(f'Checkpoint directory not found: {args.savepth}')
        args.weightpth = os.path.join(args.savepth, 'weight-final.pth')
    else:
        if args.matmcl_ckpt is None:
            raise ValueError('--matmcl_ckpt is required for matmcl_* prefixes')
        args.weightpth = args.matmcl_ckpt

    args.logger  = create_logger(args.savepth, f'{args.log}-decoder')
    args.setting = setting
    args.logger.info(print_arg(args))
    return args


def build_extractor(args, device):
    if args.prefix == 'matmcl_imagenet':
        extractor = SGPTCompatImageNet(cond_dim=args.cond_dim, cardinality=args.cardin)
    elif args.prefix == 'matmcl_clip':
        extractor = SGPTCompatCLIP(cond_dim=args.cond_dim, cardinality=args.cardin)
    else:
        config_sgpt = config_dict[args.config]
        extractor = OrderModel(
            cond_dim=args.cond_dim,
            hidden_dim=config_sgpt["hidden_dim"],
            common_dim=config_sgpt["common_dim"],
            latent_dim=config_sgpt["latent_dim"],
            dropout=config_sgpt["dropout"],
            backbone=args.backbone,
            lora_r=args.r,
            cardinality=args.cardin
        )
    extractor.load_state_dict(torch.load(args.weightpth, map_location='cpu'), strict=False)
    extractor.eval()
    return extractor.to(device)


class CachedCompositeImageDataset(Dataset):
    def __init__(self, sample_ids, image_dir, img_embs):
        self.sample_ids = sample_ids
        self.image_dir  = image_dir
        self.img_embs   = img_embs
        self.transform  = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid   = self.sample_ids[idx]
        path  = os.path.join(self.image_dir, f'{sid:03d}.png')
        image = self.transform(Image.open(path).convert('RGB'))
        return image, self.img_embs[idx]


class CachedFiberImageDataset(Dataset):
    def __init__(self, img_paths, img_embs):
        self.img_paths = img_paths
        self.img_embs  = img_embs
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.img_paths[idx]).convert('RGB'))
        return image, self.img_embs[idx]


def extract_and_cache_decoder(args, device):
    cache_path = os.path.join(args.savepth, 'emb_cache_decoder.pt')
    if os.path.exists(cache_path):
        args.logger.info(f'Loading decoder embedding cache from {cache_path}')
        return torch.load(cache_path, map_location='cpu')

    args.logger.info('Building decoder embedding cache (one-time encoder pass) …')
    extractor = build_extractor(args, device)

    if args.dataset == 'composite':
        ss        = args.split_suffix
        rootpath  = '../datasets_composite'
        image_dir = os.path.join(rootpath, 'processed')
        fea_col   = ['NumFibers', 'MMA', 'Vf', 'A11', 'A12', 'A13', 'A22', 'A23', 'A33']
        tar_col   = ['Yield strength', 'Elongation']

        src_ds1 = CompositeImageDataset('train_clean',      rootpath, image_dir,
                                        feature_cols=fea_col, target_cols=tar_col)
        src_ds2 = CompositeImageDataset('train_pred_clean', rootpath, image_dir,
                                        feature_cols=fea_col, target_cols=tar_col)
        sample_ids = src_ds1.sample_ids.tolist() + src_ds2.sample_ids.tolist()
        combined_ds = ConcatDataset([src_ds1, src_ds2])
        args.logger.info(
            f'Composite decoder: {len(src_ds1)} (train_clean) + {len(src_ds2)} '
            f'(train_pred_clean) = {len(combined_ds)} images')

        loader = DataLoader(combined_ds, batch_size=32, shuffle=False,
                            num_workers=args.n_threads)
        all_embs = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc='Extracting composite image embs'):
                all_embs.append(extractor.encode(images.to(device), 'image').cpu())
        cache = {
            'sample_ids': sample_ids,
            'img_embs':   torch.cat(all_embs),
        }

    else:  # fiber
        # Use only images from training sample IDs (preprocessed dir) to avoid leakage
        # from images/gen which may contain test samples.
        # Data always uses _clean CSVs.
        df1 = pd.read_csv('../datasets_fiber/table/mech/train_clean.csv')
        df2 = pd.read_csv('../datasets_fiber/table/mech/train_pred_clean.csv')
        train_ids = df1['ID'].tolist() + df2['ID'].tolist()
        preproc_dir = '../datasets_fiber/images/preprocessed'
        img_paths = []
        for sid in train_ids:
            sid_dir = os.path.join(preproc_dir, str(sid))
            if os.path.isdir(sid_dir):
                for fn in sorted(os.listdir(sid_dir)):
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_paths.append(os.path.join(sid_dir, fn))
        args.logger.info(
            f'Fiber decoder: {len(train_ids)} training samples → {len(img_paths)} images '
            f'from images/preprocessed (no gen-dir leakage)')
        fixed_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        all_embs = []
        batch_imgs = []
        with torch.no_grad():
            for path in tqdm(img_paths, desc='Extracting fiber preprocessed image embs'):
                img = fixed_transform(Image.open(path).convert('RGB'))
                batch_imgs.append(img)
                if len(batch_imgs) == 32:
                    batch = torch.stack(batch_imgs).to(device)
                    all_embs.append(extractor.encode(batch, 'image').cpu())
                    batch_imgs = []
            if batch_imgs:
                batch = torch.stack(batch_imgs).to(device)
                all_embs.append(extractor.encode(batch, 'image').cpu())
        cache = {
            'img_paths': img_paths,
            'img_embs':  torch.cat(all_embs),
        }

    torch.save(cache, cache_path)
    args.logger.info(f'Decoder embedding cache saved to {cache_path}')
    del extractor
    torch.cuda.empty_cache()
    return cache


def build_decoder(args, device):
    config_decoder = config_dict['decoder']
    unet = Unet(
        dim=config_decoder['dim'],
        image_embed_dim=config_decoder['image_embed_dim'],
        cond_dim=config_decoder['cond_dim'],
        channels=config_decoder['channels'],
        dim_mults=config_decoder['dim_mults']
    ).to(device)
    decoder = Decoder(
        unet=unet,
        image_size=config_decoder['image_size'],
        timesteps=config_decoder['timesteps'],
        image_cond_drop_prob=config_decoder['image_cond_drop_prob'],
        text_cond_drop_prob=config_decoder['text_cond_drop_prob'],
        learned_variance=False
    ).to(device)
    return decoder


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == 'composite':
        args.cardin   = []
        args.cond_dim = 9
    else:
        args.cardin   = [2]
        args.cond_dim = 7

    args.logger.info(f'n_epochs={args.n_epochs}  dataset={args.dataset}')

    cache = extract_and_cache_decoder(args, device)

    if args.dataset == 'composite':
        image_dir = '../datasets_composite/processed'
        train_set = CachedCompositeImageDataset(
            sample_ids=cache['sample_ids'],
            image_dir=image_dir,
            img_embs=cache['img_embs'],
        )
    else:
        train_set = CachedFiberImageDataset(
            img_paths=cache['img_paths'],
            img_embs=cache['img_embs'],
        )

    loader = DataLoader(train_set, batch_size=4, shuffle=True,
                        num_workers=args.n_threads, drop_last=True)

    decoder = build_decoder(args, device)
    decoder_trainer = DecoderTrainer(
        decoder,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).to(device)

    decoder.train()
    step = 0
    batch_loss_accumulator = []

    for epoch in range(args.n_epochs):
        args.logger.info(f'Epoch {epoch}')
        for images, img_embs in loader:
            images   = images.to(device)
            img_embs = img_embs.to(device)

            loss = decoder_trainer(images, image_embed=img_embs)
            decoder_trainer.update(1)

            batch_loss_accumulator.append(loss)
            step += 1

            if step % 1000 == 0:
                avg = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                args.logger.info(f"Step {step}: Average Loss = {avg:.4f}")
                batch_loss_accumulator = []

            if step % 2000 == 0:
                sample = decoder_trainer.sample(image_embed=img_embs[:1], one_unet_in_gpu_at_time=False)
                save_image(sample, os.path.join(args.savepth, f'gen_{epoch}_{step}.png'))

                ckpt_path = os.path.join(args.savepth, "decoder.pth")
                decoder_trainer.save(ckpt_path)

    ckpt_path = os.path.join(args.savepth, "decoder.pth")
    decoder_trainer.save(ckpt_path)
    args.logger.info(f'Decoder saved to {ckpt_path}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
