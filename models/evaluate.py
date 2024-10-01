import torch
from tqdm import tqdm
from models.utils import *

import wandb

def evaluate_clip(clip_wrapper, validation_loader, step):
    all_image_emb, all_text_emb = torch.tensor([]), torch.tensor([])
    for _, images, tokens, _ in tqdm(validation_loader):
        val_batch = (images.cuda(), tokens.cuda())
        image_emb, text_emb = clip_wrapper.calculate_emb(val_batch)
        all_image_emb = torch.cat([all_image_emb, image_emb.cpu()])
        all_text_emb = torch.cat([all_text_emb, text_emb.cpu()])

    with torch.no_grad():
        all_image_logits = all_image_emb @ all_text_emb.T
        all_text_logits = all_image_logits.T

        val_i2t_top1 = accuracy(all_image_logits)
        val_t2i_top1 = accuracy(all_text_logits)
        
        val_i2t_top5 = accuracy(all_image_logits, topk=5)
        val_t2i_top5 = accuracy(all_text_logits, topk=5)
        
        val_i2t_top10 = accuracy(all_image_logits, topk=10)
        val_t2i_top10 = accuracy(all_text_logits, topk=10)
        
        val_i2t_top20 = accuracy(all_image_logits, topk=20)
        val_t2i_top20 = accuracy(all_text_logits, topk=20)

        val_i2t_top50 = accuracy(all_image_logits, topk=50)
        val_t2i_top50 = accuracy(all_text_logits, topk=50)

        val_i2t_top200 = accuracy(all_image_logits, topk=200)
        val_t2i_top200 = accuracy(all_text_logits, topk=200)

        print(f'Top1: i2t {val_i2t_top1}, t2i {val_t2i_top1}')
        print(f'Top5: i2t {val_i2t_top5}, t2i {val_t2i_top5}')
        print(f'Top10: i2t {val_i2t_top10}, t2i {val_t2i_top10}')
        print(f'Top20: i2t {val_i2t_top20}, t2i {val_t2i_top20}')
        print(f'Top50: i2t {val_i2t_top50}, t2i {val_t2i_top50}')
        print(f'Top200: i2t {val_i2t_top200}, t2i {val_t2i_top200}')

        loss = clip_loss(all_image_logits, all_text_logits)
        
        wandb.log({'val/total_loss': loss}, step=step)
        wandb.log({
            'val/i2t_top1': val_i2t_top1,
            'val/t2i_top1': val_t2i_top1,
            'val/i2t_top5': val_i2t_top5,
            'val/t2i_top5': val_t2i_top5,
            'val/i2t_top10': val_i2t_top10,
            'val/t2i_top10': val_t2i_top10,
            'val/i2t_top50': val_i2t_top50,
            'val/t2i_top50': val_t2i_top50,
            'val/i2t_top200': val_i2t_top200,
            'val/t2i_top200': val_t2i_top200,
        }, step=step)
    return loss, (val_i2t_top1 + val_t2i_top1)/2, (val_i2t_top50 + val_t2i_top50)/2,  (val_i2t_top200 + val_t2i_top200)/2
