import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer, Unet, Decoder, DecoderTrainer
from dalle2_pytorch.dalle2_pytorch import resize_image_to
from tqdm import tqdm
from torchvision.utils import save_image
import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.model_config import config_dict
from src.models.order import OrderModel
from src.utils import set_random_seed
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


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
    parser.add_argument("--log",           type=str,   default='log')
    parser.add_argument("--device",        type=str,   default="cuda:2")
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
    parser.add_argument("--split",         type=str,   default='train',
                        choices=['train', 'test'])
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

    args.logger  = create_logger(args.savepth, f'{args.log}-gen{args.split}')
    args.setting = setting
    args.logger.info(print_arg(args))
    return args


def get_model(args, device):
    config_prior   = config_dict['prior']
    config_decoder = config_dict['decoder']

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
    extractor = extractor.to(device)

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

    return extractor, diffusion_prior, decoder


def main(args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ss = args.split_suffix

    if args.dataset == 'composite':
        feature_cols = ['NumFibers', 'MMA', 'Vf', 'A11', 'A12', 'A13', 'A22', 'A23', 'A33']
        target_cols  = ['Yield strength', 'Elongation']
        cate_cols    = None
        idx_col      = 'Image index'
        image_dir    = '../datasets_composite/processed'
        dataset_cls  = MultiModalCompositeDataset
        cardin       = []
        trainfile = '../datasets_composite/train_clean.csv'
        testfile  = '../datasets_composite/test_clean.csv'
        transform_train = transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        use_normalize = (args.prefix in ('matmcl_imagenet', 'matmcl_clip') or
                         args.backbone == 'ViT-B/16')

    elif args.dataset == 'fiber':
        feature_cols = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        target_cols  = ['fracture', 'elongation', 'elastic modulus', 'tangent modulus', 'yield']
        cate_cols    = ['dir']
        idx_col      = 'ID'
        image_dir    = '../datasets_fiber/images/preprocessed'
        dataset_cls  = MultiModalFibreDataset
        cardin       = [2]
        trainfile = '../datasets_fiber/table/mech/train_clean.csv'
        testfile  = '../datasets_fiber/table/mech/test_clean.csv'
        transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        transform_test  = transforms.Compose([transforms.Resize((224, 224)), Rotate90(), transforms.ToTensor()])
        use_normalize = True

    else:
        raise NotImplementedError(f'dataset {args.dataset} unknown')

    args.cardin   = cardin
    args.cond_dim = len(feature_cols)

    train_set = dataset_cls(
        csv_file=trainfile, image_dir=image_dir,
        feature_cols=feature_cols, target_cols=target_cols,
        extracted_fea=None, istrain=True,
        train_transform=transform_train, test_transform=transform_test,
        idx_col=idx_col, scaler=None, category_cols=cate_cols,
        use_normalize=use_normalize)
    test_set = dataset_cls(
        csv_file=testfile, image_dir=image_dir,
        feature_cols=feature_cols, target_cols=target_cols,
        extracted_fea=None, istrain=False,
        train_transform=transform_train, test_transform=transform_test,
        idx_col=idx_col, scaler=train_set.scaler, category_cols=cate_cols,
        use_normalize=use_normalize)

    extractor, diffusion_prior, decoder = get_model(args, device)
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).to(device)
    decoder_trainer = DecoderTrainer(
        decoder,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).to(device)

    prior_path   = os.path.join(args.savepth, "prior.pth")
    decoder_path = os.path.join(args.savepth, "decoder.pth")
    diffusion_prior_trainer.load(prior_path)
    decoder_trainer.load(decoder_path)
    diffusion_prior_trainer.to(device)
    decoder_trainer.to(device)

    N      = 5
    source = train_set if args.split == 'train' else test_set
    loader = DataLoader(source, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (_, _, batched_table, batched_images) in enumerate(loader):
            batched_table  = batched_table.to(device)
            batched_images = batched_images.to(device)

            table_emb = extractor.encode(batched_table, 'tab').detach()

            output_dir = os.path.join(args.savepth, f'gen-{args.split}', str(idx))
            os.makedirs(output_dir, exist_ok=True)
            args.logger.info(output_dir)
            save_image(batched_images, os.path.join(output_dir, 'real.png'))

            gen_embs = []
            for i in range(N):
                img_emb = diffusion_prior_trainer.p_sample_loop(
                    shape=(table_emb.shape[0], config_dict['prior']["image_embed_dim"]),
                    text_cond=dict(text_embed=table_emb),
                    cond_scale=1.0
                )
                gen_embs.append(img_emb)
            gen_embs = torch.cat(gen_embs, 0)

            for i in range(N):
                cur_emb = gen_embs[i].unsqueeze(0)
                gen_img = decoder_trainer.sample(image_embed=cur_emb, one_unet_in_gpu_at_time=False)
                output_path = os.path.join(output_dir, f"{idx}_{i:02d}.png")
                save_image(gen_img, output_path)


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
