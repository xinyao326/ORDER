
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import transforms
import sys
sys.path.append('..')
from src.models import *
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import *
from src.utils import *
from src.data.multimodal_dataset import MultiModalCompositeDataset, MultiModalFibreDataset
from src.model_config import config_dict
from src.trainer.loss import *

import warnings
warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--config", type=str, default='order')
    parser.add_argument("--dataset", type=str, default='composite', choices=['composite', 'fiber'])

    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_threads", type=int, default=8)

    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--backbone", type=str, default='CLIP_ViT-B/16')
    parser.add_argument("--input_split_suffix", type=str, default='_clean',
                        help="Suffix of the input split CSV files BEFORE _surr is appended. "
                             "Default '_clean' => reads train_clean_surr.csv.")
    args = parser.parse_args()

    setting = f'seed{args.seed}_surr_{args.dataset}'
    args.savepth = os.path.join('save', 'order_dyn', args.backbone.replace('/', '_'), setting)
    if not os.path.exists(args.savepth):
        os.makedirs(args.savepth)
    args.logger = create_logger(args.savepth, f'{args.log}-{args.mode}')
    args.weightpth = os.path.join(args.savepth, 'weight')
    args.setting = setting
    args.logger.info(print_arg(args))
    return args


def main(args):
    config_sgpt = config_dict[args.config]
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    surr_suffix = f'{args.input_split_suffix}_surr'

    if args.dataset == 'composite':
        feature_cols = ['NumFibers', 'MMA', 'Vf', 'A11', 'A12', 'A13', 'A22', 'A23', 'A33']
        target_cols = ['surrogate_score']
        cate_cols = None
        idx_col = 'Image index'
        image_dir = '../datasets_composite/processed'
        trainfile = f'../datasets_composite/train{surr_suffix}.csv'
        testfile  = f'../datasets_composite/test{surr_suffix}.csv'
        valfile   = f'../datasets_composite/val{surr_suffix}.csv'
        transform_train, transform_test = None, None
        dataset_cls = MultiModalCompositeDataset
        cardin = []
        use_normalize = False
        if args.backbone == 'ViT-B/16':
            use_normalize = True
    elif args.dataset == 'fiber':
        feature_cols = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        target_cols = ['surrogate_score', 'surrogate_elongation']
        cate_cols = ['dir']
        idx_col = 'ID'
        image_dir = '../datasets_fiber/images/preprocessed'
        trainfile = f'../datasets_fiber/table/mech/train{surr_suffix}.csv'
        testfile  = f'../datasets_fiber/table/mech/test{surr_suffix}.csv'
        valfile   = f'../datasets_fiber/table/mech/val{surr_suffix}.csv'
        dataset_cls = MultiModalFibreDataset
        cardin = [2]
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            Rotate90(),
            transforms.ToTensor(),
        ])
        use_normalize = True
    else:
        raise NotImplementedError(f'dataset {args.dataset} unknown')

    train_dataset = dataset_cls(
        csv_file=trainfile, image_dir=image_dir, feature_cols=feature_cols,
        target_cols=target_cols, extracted_fea=None, istrain=True,
        train_transform=transform_train, test_transform=transform_test,
        idx_col=idx_col, scaler=None, category_cols=cate_cols, use_normalize=use_normalize)
    test_dataset = dataset_cls(
        csv_file=testfile, image_dir=image_dir, feature_cols=feature_cols,
        target_cols=target_cols, extracted_fea=None, istrain=False,
        train_transform=transform_train, test_transform=transform_test,
        idx_col=idx_col, scaler=train_dataset.scaler, category_cols=cate_cols,
        use_normalize=use_normalize)
    val_dataset = dataset_cls(
        csv_file=valfile, image_dir=image_dir, feature_cols=feature_cols,
        target_cols=target_cols, extracted_fea=None, istrain=False,
        train_transform=transform_train, test_transform=transform_test,
        idx_col=idx_col, scaler=train_dataset.scaler, category_cols=cate_cols,
        use_normalize=use_normalize)

    cond_dim = len(feature_cols)

    args.logger.info(f"train:{len(train_dataset)}, val:{len(val_dataset)}, test:{len(test_dataset)}")
    args.logger.info(f"Surrogate target cols: {target_cols}")
    args.logger.info(f"Surrogate mean: {train_dataset.mean}, std: {train_dataset.std}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_threads, worker_init_fn=seed_worker,
                              generator=g, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.n_threads, worker_init_fn=seed_worker,
                              generator=g, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_threads, worker_init_fn=seed_worker,
                              generator=g, drop_last=True)

    model_cls = OrderModel
    loss_fn_clip = cliploss
    loss_fn_rnc = RnCLoss(
        train_dataset.mean, train_dataset.std, args.device, args.n_epochs,
        label_diff='l2', feature_sim='product')
    loss_fn = {'clip': loss_fn_clip, 'rnc': loss_fn_rnc}

    if cate_cols is not None:
        assert len(cardin) == len(train_dataset.category_cols)

    model = model_cls(
        cond_dim=cond_dim,
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=args.backbone,
        lora_r=args.r,
        cardinality=cardin,
    ).to(device)

    if args.mode == 'train':
        args.logger.info('training')
        optimizer = Adam(model.parameters(), lr=config_sgpt["lr"],
                         weight_decay=config_sgpt["weight_decay"])
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup_updates=args.n_epochs * len(train_dataset) // 32 // 10,
            tot_updates=args.n_epochs * len(train_dataset) // 32,
            lr=config_sgpt["lr"], end_lr=1e-9, power=1)
        trainer = DynTrainer(args, optimizer, lr_scheduler, loss_fn, device=device)
        train_result, test_result = trainer.fit(model, train_loader, val_loader, test_loader)
        args.logger.info(f"Train loss: {train_result}\tTest loss: {test_result}")
    else:
        args.logger.info('testing')
        ks = [1, 3, 5, 10]
        model.load_state_dict(torch.load(f'{args.weightpth}-final.pth'), strict=False)
        model.eval()
        loader = test_loader
        tab_rep, img_rep, all_tar = None, None, None
        with torch.no_grad():
            for batched_data in loader:
                (idx, targets, x_tabular, x_img) = batched_data
                x_tabular = x_tabular.to(device)
                x_img = x_img.to(device)
                batch_repr = model.forward_unsupervised(x_tabular, x_img)
                tab_rep = batch_repr[0] if tab_rep is None else torch.concat([tab_rep, batch_repr[0]])
                img_rep = batch_repr[1] if img_rep is None else torch.concat([img_rep, batch_repr[1]])
                all_tar = targets if all_tar is None else torch.concat([all_tar, targets], 0)
        for kk in ks:
            result = compute_retrieval(img_rep, tab_rep, k=kk)
            args.logger.info(result)

        sort_idx = torch.argsort(all_tar[:, 0])
        img_sorted = img_rep[sort_idx].cpu()
        tab_sorted = tab_rep[sort_idx].cpu()
        sim_pth = os.path.join(args.savepth, 'simmat_test_sorted.png')
        plot_similarity_matrix(img_sorted, tab_sorted, save_path=sim_pth)
        args.logger.info(f'Saved sorted similarity matrix → {sim_pth}')

        tsne_pth = os.path.join(args.savepth, 'tsne_test.png')
        unified_analyze_new(img_rep.cpu(), tab_rep.cpu(), all_tar[:, 0].cpu(), pth=tsne_pth)
        args.logger.info(f'Saved t-SNE → {tsne_pth}')


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
