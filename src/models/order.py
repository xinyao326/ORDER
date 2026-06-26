from torch import nn
import torch.nn.functional as F

from src.models import TableTransformerWrapper
from src.models.vit import ViT32, ViT16
from src.models.myclip import *


class OrderModel(nn.Module):
    def __init__(self, cond_dim, hidden_dim, common_dim, latent_dim, dropout=0, backbone="cnn", lora_r=8, cardinality=[2]) -> None:
        super(OrderModel, self).__init__()
        if backbone == "ViT-B/32":
            self.table_processor = TableTransformerWrapper(in_dim=cond_dim, out_dim=common_dim, dropout=dropout, cardinality=cardinality)
            self.vision_processor = ViT32(out_dim=common_dim)
        elif backbone == 'ViT-B/16':
            self.table_processor = TableTransformerWrapper(in_dim=cond_dim, out_dim=common_dim, dropout=dropout, cardinality=cardinality)
            self.vision_processor = ViT16(out_dim=common_dim)
        elif 'CLIP' in backbone:
            model_name_dict = {
                'CLIP_ViT-B/32': 'openai/clip-vit-base-patch32',
                'CLIP_ViT-B/16': 'openai/clip-vit-base-patch16',
                'CLIP_ViT-L/14': 'openai/clip-vit-large-patch14',
                'CLIP_ViT-L/14-336': 'openai/clip-vit-large-patch14-336',
            }
            self.table_processor = TableTransformerWrapper(in_dim=cond_dim, out_dim=common_dim, dropout=dropout, cardinality=cardinality)
            self.vision_processor = peftCLIP(model_name=model_name_dict[backbone], output_dim=common_dim, lora_r=lora_r)
        else:
            raise ValueError(f"Backbone {backbone} unknown")

        self.encoder = Encoder(common_dim=common_dim, latent_dim=latent_dim)

    def forward_unsupervised(self, x_table, x_img):
        tab_rep = self.encoder(self.table_processor(x_table, x_img))
        vis_rep = self.encoder(self.vision_processor(x_table, x_img))
        return [tab_rep, vis_rep]

    def encode_table_repr(self, x_table, _):
        table_representation = self.encoder(self.table_processor(x_table, _))
        return table_representation

    def encode_image_repr(self, _, x_img):
        image_representation = self.encoder(self.vision_processor(_, x_img))
        return image_representation
    
    def encode(self, x, modal):
        if modal == 'tab':
            return self.encode_table_repr(x, x)
        elif modal == 'image':
            return self.encode_image_repr(x, x)
        else:
            raise RuntimeError
    

class Encoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(Encoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)


if __name__ == "__main__":
    pass