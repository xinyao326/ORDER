import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

sys.path.append('..')
from src.data.multimodal_dataset import MultiModalCompositeDataset, MultiModalFibreDataset
from src.model_config import config_dict
from src.models.order import OrderModel
from src.trainer.evaluator import Evaluator
from src.trainer.finetune_trainer import Trainer
from src.trainer.result_tracker import Result_Tracker
from src.trainer.scheduler import PolynomialDecayLR
from src.utils import create_logger, set_random_seed, print_arg, Rotate90

from torchvision.models import vit_b_16, ViT_B_16_Weights

MATMCL_IMAGENET_PATH = ''
MATMCL_CLIP_PATH     = ''


def _evict_src_modules():
    for key in list(sys.modules.keys()):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]


def _load_sgpt_imagenet(ckpt_path, dataset, device):
    import importlib
    _evict_src_modules()
    sys.path.insert(0, os.path.join(MATMCL_IMAGENET_PATH, 'scipts'))
    sys.path.insert(0, MATMCL_IMAGENET_PATH)
    sgpt_mod = importlib.import_module('src.models.sgpt')
    cfg_mod  = importlib.import_module('src.model_config')
    SGPT     = sgpt_mod.SGPT
    cfg_dict = cfg_mod.config_dict

    if dataset == 'fiber':
        cfg  = cfg_dict['sgpt_vit16']
        sgpt = SGPT(cond_dim=cfg['cond_dim'], hidden_dim=cfg['hidden_dim'],
                    common_dim=cfg['common_dim'], latent_dim=cfg['latent_dim'],
                    dropout=cfg['dropout'], backbone=cfg['backbone']).to(device)
    else:
        cfg  = cfg_dict['sgpt_composite_vit16']
        sgpt = SGPT(cond_dim=cfg['cond_dim'], hidden_dim=cfg['hidden_dim'],
                    common_dim=cfg['common_dim'], latent_dim=cfg['latent_dim'],
                    dropout=cfg['dropout'], backbone=cfg['backbone']).to(device)

    sgpt.load_state_dict(torch.load(ckpt_path, map_location=device))
    for p in [os.path.join(MATMCL_IMAGENET_PATH, 'scipts'), MATMCL_IMAGENET_PATH]:
        while p in sys.path:
            sys.path.remove(p)
    _evict_src_modules()
    return sgpt


def _load_sgpt_clip(ckpt_path, dataset, device):
    import importlib
    _evict_src_modules()
    sys.path.insert(0, MATMCL_CLIP_PATH)
    sgpt_mod = importlib.import_module('src.models.sgpt')
    cfg_mod  = importlib.import_module('src.model_config')
    SGPT     = sgpt_mod.SGPT
    cfg_dict = cfg_mod.config_dict

    if dataset == 'fiber':
        cfg  = cfg_dict['sgpt_clip_fiber']
        sgpt = SGPT(cond_dim=cfg['cond_dim'], hidden_dim=cfg['hidden_dim'],
                    common_dim=cfg['common_dim'], latent_dim=cfg['latent_dim'],
                    dropout=cfg['dropout'], backbone=cfg['backbone'],
                    uselora=True, lora_r=8, cardinality=[2]).to(device)
    else:
        cfg  = cfg_dict['sgpt_clip_composite']
        sgpt = SGPT(cond_dim=cfg['cond_dim'], hidden_dim=cfg['hidden_dim'],
                    common_dim=cfg['common_dim'], latent_dim=cfg['latent_dim'],
                    dropout=cfg['dropout'], backbone=cfg['backbone'],
                    uselora=True, lora_r=8, cardinality=[]).to(device)

    sgpt.load_state_dict(torch.load(ckpt_path, map_location=device))
    while MATMCL_CLIP_PATH in sys.path:
        sys.path.remove(MATMCL_CLIP_PATH)
    _evict_src_modules()
    return sgpt


def build_mlp_head(in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.2):
    layers = [nn.Linear(in_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def extract_fusion_features_order(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for batch in loader:
        idx, targets, x_tab, x_img = batch
        x_tab, x_img = x_tab.to(device), x_img.to(device)
        tab_f = encoder.encode(x_tab, 'tab')
        img_f = encoder.encode(x_img, 'image')
        feats.append(((tab_f + img_f) / 2).cpu())
        labels.append(targets)
    return torch.cat(feats, 0), torch.cat(labels, 0)


@torch.no_grad()
def extract_fusion_features_matmcl(sgpt, loader, device):
    sgpt.eval()
    feats, labels = [], []
    for batch in loader:
        idx, targets, x_tab, x_img = batch
        x_tab, x_img = x_tab.to(device), x_img.to(device)
        feats.append(sgpt.encode_fusion_repr(x_tab, x_img).cpu())
        labels.append(targets)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--n_epochs",     type=int,   default=50)
    parser.add_argument("--dataset",      type=str,   default='composite',
                        choices=['composite', 'fiber'])
    parser.add_argument("--metric",       type=str,   default='rmse')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--split_suffix", type=str,   default='_clean')
    parser.add_argument("--log",          type=str,   default='log')

    parser.add_argument("--prefix",       type=str,   default='order_dyn',
                        choices=['order_dyn', 'order_alpha', 'matmcl_imagenet', 'matmcl_clip',
                                 'baseline_dwcl', 'baseline_triplet'])
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--backbone",     type=str,   default='CLIP_ViT-B/16')
    parser.add_argument("--r",            type=int,   default=8)
    parser.add_argument("--config",       type=str,   default='order')

    parser.add_argument("--matmcl_ckpt",  type=str,   default=None)

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
    elif args.prefix in ('baseline_dwcl', 'baseline_triplet'):
        loss_type = args.prefix.split('_', 1)[1]  # 'dwcl' or 'triplet'
        setting = f'seed{args.seed}_{loss_type}_Alpha{args.alpha}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix,
                                    args.backbone.replace('/', '_'), setting)
    else:
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix, setting)
        os.makedirs(args.savepth, exist_ok=True)

    if args.prefix in ('order_dyn', 'order_alpha', 'baseline_dwcl', 'baseline_triplet'):
        if not os.path.exists(args.savepth):
            raise RuntimeError(f'Checkpoint directory not found: {args.savepth}')
        args.weightpth = os.path.join(args.savepth, 'weight-final.pth')
    else:
        if args.matmcl_ckpt is None:
            raise ValueError('--matmcl_ckpt required for matmcl_* prefixes')
        args.weightpth = args.matmcl_ckpt

    args.logger = create_logger(args.savepth, f'{args.log}-predict_fusion')
    args.logger.info(print_arg(args))
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ss = args.split_suffix

    use_norm = (args.prefix in ('matmcl_imagenet', 'matmcl_clip') or
                args.backbone == 'ViT-B/16')

    if args.dataset == 'composite':
        fea_col  = ['NumFibers', 'MMA', 'Vf', 'A11', 'A12', 'A13', 'A22', 'A23', 'A33']
        tar_col  = ['Yield strength', 'Elongation']
        cardin   = []
        cond_dim = len(fea_col)
        n_task   = 2
        image_dir  = '../datasets_composite/processed'
        trainfile  = f'../datasets_composite/train_pred{ss}.csv'
        testfile   = f'../datasets_composite/test_pred{ss}.csv'
        valfile    = f'../datasets_composite/val_pred{ss}.csv'
        dataset_cls = MultiModalCompositeDataset
        if args.prefix in ('matmcl_imagenet', 'matmcl_clip'):
            _t_no_norm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            comp_train_tf, comp_test_tf = _t_no_norm, _t_no_norm
        else:
            comp_train_tf, comp_test_tf = None, None  # ORDER uses dataset defaults
        kwargs = dict(feature_cols=fea_col, target_cols=tar_col, idx_col='Image index',
                      category_cols=None, use_normalize=use_norm,
                      train_transform=comp_train_tf, test_transform=comp_test_tf, extracted_fea=None)
    else:  # fiber
        fea_col  = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        tar_col  = ['fracture', 'elongation', 'elastic modulus', 'tangent modulus', 'yield']
        cardin   = [2]
        cond_dim = len(fea_col)
        n_task   = 5
        image_dir  = '../datasets_fiber/images/preprocessed'
        trainfile  = f'../datasets_fiber/table/mech/train_pred{ss}.csv'
        testfile   = f'../datasets_fiber/table/mech/test_pred{ss}.csv'
        valfile    = f'../datasets_fiber/table/mech/val_pred{ss}.csv'
        dataset_cls = MultiModalFibreDataset
        t_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
        ])
        t_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            Rotate90(),
            transforms.ToTensor(),
        ])
        kwargs = dict(feature_cols=fea_col, target_cols=tar_col, idx_col='ID',
                      category_cols=['dir'], use_normalize=True,
                      train_transform=t_train, test_transform=t_test, extracted_fea=None)

    train_set = dataset_cls(csv_file=trainfile, image_dir=image_dir, istrain=True,
                             scaler=None, **kwargs)
    val_set   = dataset_cls(csv_file=valfile,   image_dir=image_dir, istrain=False,
                             scaler=train_set.scaler, **kwargs)
    test_set  = dataset_cls(csv_file=testfile,  image_dir=image_dir, istrain=False,
                             scaler=train_set.scaler, **kwargs)

    train_mean = np.array(train_set.mean)
    train_std  = np.array(train_set.std)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

    cfg = config_dict[args.config]
    args.logger.info('Pre-extracting fusion features ...')

    if args.prefix in ('order_dyn', 'order_alpha', 'baseline_dwcl', 'baseline_triplet'):
        encoder = OrderModel(
            cond_dim=cond_dim,
            hidden_dim=cfg['hidden_dim'], common_dim=cfg['common_dim'],
            latent_dim=cfg['latent_dim'], dropout=cfg['dropout'],
            backbone=args.backbone, lora_r=args.r, cardinality=cardin,
        ).to(device)
        encoder.load_state_dict(
            torch.load(args.weightpth, map_location=device), strict=False)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        tr_feat, tr_y = extract_fusion_features_order(encoder, train_loader, device)
        va_feat, va_y = extract_fusion_features_order(encoder, val_loader,   device)
        te_feat, te_y = extract_fusion_features_order(encoder, test_loader,  device)

    elif args.prefix == 'matmcl_imagenet':
        sgpt = _load_sgpt_imagenet(args.weightpth, args.dataset, device)
        sgpt.eval()
        for p in sgpt.parameters():
            p.requires_grad_(False)
        tr_feat, tr_y = extract_fusion_features_matmcl(sgpt, train_loader, device)
        va_feat, va_y = extract_fusion_features_matmcl(sgpt, val_loader,   device)
        te_feat, te_y = extract_fusion_features_matmcl(sgpt, test_loader,  device)

    elif args.prefix == 'matmcl_clip':
        sgpt = _load_sgpt_clip(args.weightpth, args.dataset, device)
        sgpt.eval()
        for p in sgpt.parameters():
            p.requires_grad_(False)
        tr_feat, tr_y = extract_fusion_features_matmcl(sgpt, train_loader, device)
        va_feat, va_y = extract_fusion_features_matmcl(sgpt, val_loader,   device)
        te_feat, te_y = extract_fusion_features_matmcl(sgpt, test_loader,  device)

    args.logger.info(f'  train={tr_feat.shape}, val={va_feat.shape}, test={te_feat.shape}')

    lat_dim = tr_feat.shape[1]
    tr_loader = DataLoader(TensorDataset(tr_feat, tr_y), batch_size=32, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_feat, va_y), batch_size=32, shuffle=False)
    te_loader = DataLoader(TensorDataset(te_feat, te_y), batch_size=32, shuffle=False)

    model = build_mlp_head(lat_dim, cfg['hidden_dim'], n_task,
                           num_layers=3, dropout=args.dropout).to(device)

    optimizer    = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.n_epochs * len(tr_y) // 32 // 10,
        tot_updates=args.n_epochs * len(tr_y) // 32,
        lr=args.lr, end_lr=1e-9, power=1)

    label_mean = torch.tensor(train_mean).to(device)
    label_std  = torch.tensor(train_std).to(device)

    evaluator       = Evaluator('MLP', args.metric,  n_task, mean=train_mean, std=train_std)
    final_eval_rmse = Evaluator('MLP', 'rmse_split', n_task, mean=train_mean, std=train_std)
    final_eval_r2   = Evaluator('MLP', 'r2',         n_task, mean=train_mean, std=train_std)
    result_tracker  = Result_Tracker(args.metric)

    trainer = Trainer(args, optimizer, lr_scheduler, nn.MSELoss(),
                      evaluator, final_eval_rmse, final_eval_r2, result_tracker,
                      summary_writer=None, device=device, model_name='MLP',
                      label_mean=label_mean, label_std=label_std)

    best_train, best_val, best_test, test_final = trainer.fit(
        model, tr_loader, va_loader, te_loader)
    args.logger.info(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")

    avg_r2 = 0.0
    for i in range(n_task):
        rmse_fmt = f"{test_final[0][i]:.4f}" if args.dataset == 'composite' else f"{test_final[0][i]:.3f}"
        args.logger.info(f"{tar_col[i]}: test rmse: {rmse_fmt}"
                         f"\ttest r2: {test_final[1][i]:.3f}")
        avg_r2 += test_final[1][i]
    args.logger.info(f"Avg R2: {avg_r2 / n_task:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
