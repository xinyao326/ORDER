import torch.nn as nn
import warnings
from torch.cuda.amp import autocast
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModel
warnings.filterwarnings('ignore')


class Encoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(Encoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return nn.functional.normalize(self.encode(x), dim=-1)


class peftCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", output_dim=512, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()        
        self.clip_model = CLIPVisionModel.from_pretrained(model_name)
        self.feature_dim = self.clip_model.config.hidden_size
        
        lora_config = LoraConfig(
            r=lora_r,                         
            lora_alpha=lora_alpha,            
            target_modules=[                   
                "q_proj",                       
                "v_proj",                       
                "k_proj",                       
                "out_proj",                    
            ],
            lora_dropout=lora_dropout,        
            bias="none",                       
        )
        self.clip_model = get_peft_model(self.clip_model, lora_config)
        self.clip_model.print_trainable_parameters()
        self.fc = nn.Linear(self.feature_dim, output_dim)
    
    def forward(self, tab, images):
        with autocast():
            outputs = self.clip_model(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]  
            output = self.fc(features)
        
        return output.float()
    

