import argparse
import os
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision.models import vit_b_16, ViT_B_16_Weights
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer
from tqdm import tqdm

import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.model_config import config_dict
from src.models.order import OrderModel
from src.models import OrderModel
from src.utils import set_random_seed



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
    parser.add_argument("--n_epochs",      type=int,   default=1000)
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

    args.logger  = create_logger(args.savepth, f'{args.log}-prior')
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


def extract_and_cache_prior(args, train_set, device):
    cache_path = os.path.join(args.savepth, 'emb_cache_prior.pt')
    if os.path.exists(cache_path):
        args.logger.info(f'Loading embedding cache from {cache_path}')
        return torch.load(cache_path, map_location='cpu')

    args.logger.info('Building embedding cache (one-time pass through encoder) …')
    extractor = build_extractor(args, device)

    if args.dataset == 'composite':
        loader = DataLoader(train_set, batch_size=128, shuffle=False,
                            num_workers=args.n_threads)
        all_tab, all_img = [], []
        with torch.no_grad():
            for _, _, tab, img in tqdm(loader, desc='Extracting composite embeddings'):
                all_tab.append(extractor.encode(tab.to(device), 'tab').cpu())
                all_img.append(extractor.encode(img.to(device), 'image').cpu())
        cache = {
            'tab_embs': torch.cat(all_tab),
            'img_embs': torch.cat(all_img),
        }

    else:  # fiber
        image_dir = '../datasets_fiber/images/preprocessed'
        tab_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        rot_transform = transforms.Compose([transforms.Resize((224, 224)), Rotate90(), transforms.ToTensor()])

        all_tab_raw = torch.tensor(train_set.features, dtype=torch.float32)
        sample_ids  = train_set.sample_ids.tolist()

        tab_embs = []
        with torch.no_grad():
            for start in range(0, len(all_tab_raw), 128):
                chunk = all_tab_raw[start:start + 128].to(device)
                tab_embs.append(extractor.encode(chunk, 'tab').cpu())

        img_emb_banks = []
        with torch.no_grad():
            for i, (sid, row_feats) in enumerate(
                    tqdm(zip(sample_ids, all_tab_raw), total=len(sample_ids),
                         desc='Extracting fiber image banks')):
                direction  = int(row_feats[-1].item())
                img_dir_i  = os.path.join(image_dir, str(sid))
                fns        = sorted(os.listdir(img_dir_i))
                imgs = []
                for fn in fns:
                    img = Image.open(os.path.join(img_dir_i, fn))
                    tfm = tab_transform if direction == 1 else rot_transform
                    imgs.append(tfm(img))
                img_batch = torch.stack(imgs).to(device)
                bank = extractor.encode(img_batch, 'image').cpu()
                img_emb_banks.append(bank)

        cache = {
            'tab_embs':      torch.cat(tab_embs),
            'img_emb_banks': img_emb_banks,
        }

    torch.save(cache, cache_path)
    args.logger.info(f'Embedding cache saved to {cache_path}')
    del extractor
    torch.cuda.empty_cache()
    return cache


def build_prior(args, device):
    config_prior = config_dict['prior']
    prior_network = DiffusionPriorNetwork(
        dim=config_prior["dim"],
        depth=config_prior["depth"],
        dim_head=config_prior["dim_head"],
        heads=config_prior["heads"]
    ).to(device)
    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=config_prior["image_embed_dim"],
        timesteps=config_prior["timesteps"],
        cond_drop_prob=config_prior["cond_drop_prob"],
        condition_on_text_encodings=False
    ).to(device)
    return diffusion_prior


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ss = args.split_suffix

    if args.dataset == 'composite':
        feature_cols  = ['NumFibers', 'MMA', 'Vf', 'A11', 'A12', 'A13', 'A22', 'A23', 'A33']
        target_cols   = ['Yield strength', 'Elongation']
        idx_col       = 'Image index'
        image_dir     = '../datasets_composite/processed'
        args.cardin   = []
        args.cond_dim = len(feature_cols)

        use_norm = (args.prefix in ('matmcl_imagenet', 'matmcl_clip') or
                    args.backbone == 'ViT-B/16')

        trainfile1 = '../datasets_composite/train_clean.csv'
        trainfile2 = '../datasets_composite/train_pred_clean.csv'
        train1 = MultiModalCompositeDataset(
            csv_file=trainfile1, image_dir=image_dir,
            feature_cols=feature_cols, target_cols=target_cols,
            extracted_fea=None, istrain=True,
            train_transform=None, test_transform=None,
            idx_col=idx_col, scaler=None, category_cols=None,
            use_normalize=use_norm)
        train2 = MultiModalCompositeDataset(
            csv_file=trainfile2, image_dir=image_dir,
            feature_cols=feature_cols, target_cols=target_cols,
            extracted_fea=None, istrain=True,
            train_transform=None, test_transform=None,
            idx_col=idx_col, scaler=train1.scaler, category_cols=None,
            use_normalize=use_norm)
        train_set = ConcatDataset([train1, train2])
        args.logger.info(
            f'Composite prior training: {len(train1)} (train{ss}) + {len(train2)} '
            f'(train_pred{ss}) = {len(train_set)} samples')

    elif args.dataset == 'fiber':
        feature_cols  = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        target_cols   = ['fracture', 'elongation', 'elastic modulus', 'tangent modulus', 'yield']
        idx_col       = 'ID'
        image_dir     = '../datasets_fiber/images/preprocessed'
        args.cardin   = [2]
        args.cond_dim = len(feature_cols)

        trainfile1 = '../datasets_fiber/table/mech/train_clean.csv'
        trainfile2 = '../datasets_fiber/table/mech/train_pred_clean.csv'
        train1 = MultiModalFibreDataset(
            csv_file=trainfile1, image_dir=image_dir,
            feature_cols=feature_cols, target_cols=target_cols,
            extracted_fea=None, istrain=True,
            train_transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
            test_transform=transforms.Compose([transforms.Resize((224, 224)), Rotate90(), transforms.ToTensor()]),
            idx_col=idx_col, scaler=None, category_cols=['dir'],
            use_normalize=True)
        train2 = MultiModalFibreDataset(
            csv_file=trainfile2, image_dir=image_dir,
            feature_cols=feature_cols, target_cols=target_cols,
            extracted_fea=None, istrain=True,
            train_transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
            test_transform=transforms.Compose([transforms.Resize((224, 224)), Rotate90(), transforms.ToTensor()]),
            idx_col=idx_col, scaler=train1.scaler, category_cols=['dir'],
            use_normalize=True)
        train_set = types.SimpleNamespace(
            features   = np.vstack([train1.features, train2.features]),
            sample_ids = np.concatenate([train1.sample_ids, train2.sample_ids]),
        )
        args.logger.info(
            f'Fiber prior training: {len(train1.features)} (train{ss}) + {len(train2.features)} '
            f'(train_pred{ss}) = {len(train_set.features)} samples')

    else:
        raise NotImplementedError(f'dataset {args.dataset} unknown')

    cache = extract_and_cache_prior(args, train_set, device)

    diffusion_prior = build_prior(args, device)
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=50, 
        ema_update_every=10,
    ).to(device)

    batch_size = 32
    step = 0
    batch_loss_accumulator = []

    if args.dataset == 'composite':
        tab_embs = cache['tab_embs'].to(device)
        img_embs = cache['img_embs'].to(device)
        N = tab_embs.shape[0]

        for epoch in range(args.n_epochs):
            args.logger.info(f'Epoch {epoch}:')
            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                loss = diffusion_prior_trainer(
                    text_embed=tab_embs[idx],
                    image_embed=img_embs[idx],
                )
                diffusion_prior_trainer.update()
                batch_loss_accumulator.append(loss)
                step += 1
                if step % 10 == 0:
                    avg = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                    args.logger.info(f"Step {step}: Average Loss = {avg:.4f}")
                    batch_loss_accumulator = []

    else:  # fiber
        tab_embs      = cache['tab_embs'].to(device)
        img_emb_banks = cache['img_emb_banks']         
        N = tab_embs.shape[0]

        for epoch in range(args.n_epochs):
            args.logger.info(f'Epoch {epoch}:')
            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx_list  = perm[start:start + batch_size].tolist()
                tab_batch = tab_embs[idx_list]
                img_batch = torch.stack([
                    img_emb_banks[i][torch.randint(len(img_emb_banks[i]), (1,)).item()]
                    for i in idx_list
                ]).to(device)
                loss = diffusion_prior_trainer(
                    text_embed=tab_batch,
                    image_embed=img_batch,
                )
                diffusion_prior_trainer.update()
                batch_loss_accumulator.append(loss)
                step += 1
                if step % 10 == 0:
                    avg = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                    args.logger.info(f"Step {step}: Average Loss = {avg:.4f}")
                    batch_loss_accumulator = []

    savepth = os.path.join(args.savepth, "prior.pth")
    diffusion_prior_trainer.save(savepth)
    args.logger.info(f'Prior saved to {savepth}')


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
