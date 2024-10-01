import os
import time
import torch
import argparse
import random
import numpy as np

from models.wrapper import CLIPWrapper
from models.data import CLIPDataset, CLIPEmbeddingDataset
from models.evaluate import evaluate_clip

from torchvision.transforms import transforms

import wandb

# Log in wandb
wandb.login()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple training script for CLIP object.")

    # Add arguments
    parser.add_argument('--seed', type=int, default=None, help="Random seed.")
    parser.add_argument('--model-name', type=str, default="openai/clip-vit-base-patch32",
        help="Huggingface repository name for CLIP model.")
    parser.add_argument('--preprocessor-path', type=str, default="openai/clip-vit-base-patch32",
        help="Huggingface repository name for CLIP preprocessor.")
    parser.add_argument('--exp-name', type=str, required=True,
        help="Experiment name, for logging purpose.")
    parser.add_argument('--run-name', type=str, default=None,
        help="Run name, for logging purpose.")
    parser.add_argument('--train_tsv', type=str,
        help="Path to the train tsv file.")
    parser.add_argument('--val_tsv', type=str,
        help="Path to the test tsv file.")
    parser.add_argument('--caption_keys', type=str,
        default='caption1,caption2,caption3,caption4,caption5',
        help="Keys for caption column in train/test tsv file.")
    parser.add_argument('--train-image-dir', type=str,
        help="Path to train images.")
    parser.add_argument('--val-image-dir', type=str,
        help="Path to test images")
    parser.add_argument('--batch-size', type=int, default=1024,
        help="Train batch size.")
    parser.add_argument('--val-batch-size', type=int, default=1024,
        help="Test batch size.")
    parser.add_argument('--num-epochs', type=int, default=15,
        help="Total number of epochs to train.")
    parser.add_argument('--learning-rate', type=float, default=None,
        help="Learning rate. Default to 5e-4 for pretrain and 1e-5 for finetune.")
    parser.add_argument('--max-iters-per-epoch', type=int, default=None,
        help="Maximum number of iterations in one epoch. Default to max dataloader length.")


    parser.add_argument('--save-dir', type=str, default="./ckpts")
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--save-freq', type=int, default=10)

    parser.add_argument('--data-parallel', type=bool, default=True)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--log-offset', type=int, default=0)

   # Arguments for synthesis batch
    parser.add_argument('--synthesis', action='store_true', default=False,
        help="Activate synthesis sampling in training.")
    parser.add_argument('--preprocess-neighbors', action='store_true', default=False,
        help="Whether to preprocess the neighbor ids when calculating nearest neighbors." \
        "This is useful for large datasets such as CC3M.")
    parser.add_argument('--synthesis-prop', type=float, default=1.0,
        help="Similarity proportion p.")
    parser.add_argument('--num-clusters', type=int, default=4,
        help="Number of clusters n_clusters.")
    parser.add_argument('--online-sim', type=str, default=None,
        help="Whether to use online embeddings.")
    parser.add_argument('--neighborhood', type=int, default=1,
        help="Neighborhood size s.")

    parser.add_argument('--warmup-prop', action='store_true', default=False,
        help="Activate warmup similarity proportion.")
    parser.add_argument('--cluster-size', type=int, default=4,
        help="Cluster size k, only used for warmup proportion." \
        "For non-warmup settings, please specify `num_clusters`.")
    parser.add_argument('--upweight-epochs', type=str, default="4,6,8,10,12,14,16,18",
        help="Epochs to upweight the similarity proportion p during the training process.")

    # Arguments for mix-in experiments
    parser.add_argument('--mnist-injection', type=int, default=0,
        help="Number of mnist data to mix in.")

    # Parse arguments
    args = parser.parse_args()
    return args


def get_batch_split(batch_size, prop=0.0):
    synthesis_size = int(batch_size * prop)
    trainloader_batch_size = (1 if batch_size - synthesis_size == 0 else 
                              batch_size - synthesis_size)

    return trainloader_batch_size, synthesis_size


def batch_evaluate(step, wrapper, validation_loader):
    val_loss, val_i2t, val_t2i, total_batches = 0.0, 0.0, 0.0, 0
    for i, (image_paths, images, tokens, token_len) in enumerate(validation_loader):
        val_batch = (images.cuda(), tokens.cuda())
        batch_loss, batch_i2t, batch_t2i = wrapper.validation_step(val_batch, step)


def main():
    assert torch.cuda.is_available()
    args = parse_arguments()

    if args.seed:
        # Set global seed
        set_seed(args.seed)

    if args.online_sim:
        assert args.online_sim in ['text', 'image']

    if args.warmup_prop:
        assert args.synthesis

    if args.log_dir is None:
        if args.warmup_prop:
            if args.online_sim == "text":
                exp_name = f"warmup+text+prop1.0+size{args.cluster_size}+neighborhood{args.neighborhood}+seed{args.seed}"
            elif args.online_sim == "image":
                exp_name = f"warmup+image+prop1.0+size{args.cluster_size}+neighborhood{args.neighborhood}+seed{args.seed}"
            else:
                exp_name = f"warmup+offline+prop1.0+size{args.cluster_size}+neighborhood{args.neighborhood}+seed{args.seed}"
        elif args.synthesis:
            cluster_size = int(args.synthesis_prop * args.batch_size) // args.num_clusters
            if args.online_sim == "text":
                exp_name = f"text+prop{args.synthesis_prop}+size{cluster_size}+neighborhood{args.neighborhood}+seed{args.seed}"
            elif args.online_sim == "image":
                exp_name = f"image+prop{args.synthesis_prop}+size{cluster_size}+neighborhood{args.neighborhood}+seed{args.seed}"
            else:
                exp_name = f"offline+prop{args.synthesis_prop}+size{cluster_size}+neighborhood{args.neighborhood}+seed{args.seed}"
        else:
            exp_name = f"finetune+seed{args.seed}"
        
        args.log_dir = exp_name
        args.save_dir = os.path.join(args.save_dir, args.log_dir)
        assert not os.path.exists(args.save_dir)
    else:
        args.save_dir = os.path.join(args.save_dir, args.log_dir)
        if os.path.exists(args.save_dir):
            index = 1
            while True:
                current_dir = args.save_dir + str(index)
                if not os.path.exists(current_dir):
                    args.save_dir = current_dir
                    break
                index += 1
        exp_name = args.log_dir

    os.makedirs(args.save_dir)
    print(f'Logging to {args.save_dir}')
    if args.run_name is None:
        args.run_name = exp_name
    wandb.init(project=args.exp_name, name=args.run_name, config=vars(args))

    args.caption_keys = list(args.caption_keys.split(','))
    if args.synthesis:
        train_dataset = CLIPEmbeddingDataset(
            args.train_tsv, args.train_image_dir,
            'image', args.caption_keys,
            args.preprocessor_path,
            transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            ]), random_caption=True)

    else:
        train_dataset = CLIPDataset(
            args.train_tsv, args.train_image_dir,
            'image', args.caption_keys,
            args.preprocessor_path,
            transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            ]), random_caption=True)

    validation_dataset = CLIPDataset(args.val_tsv, args.val_image_dir,
                                     'image', args.caption_keys,
                                     args.preprocessor_path, random_caption=False)

    if args.mnist_injection > 0:
        train_dataset.inject_mnist(args.mnist_injection)

    print(f'Training samples: {len(train_dataset)}, validation samples: {len(validation_dataset)}')

    if args.warmup_prop:
        print("Using warm up prop scheduling.")
        upweight_epochs = list(args.upweight_epochs.split(','))
        upweight_epochs = [int(e) for e in upweight_epochs]
        args.num_clusters = 1  # Initially set for warmup
    else:
        upweight_epochs = []

    if args.synthesis:
        print('Activating semi-batch synthesis.')
        trainloader_batch_size, synthesis_size = get_batch_split(args.batch_size, args.synthesis_prop)
        assert synthesis_size >= 0 and trainloader_batch_size >= 0
        print(f'Using {synthesis_size} categorized and {args.batch_size - synthesis_size} random samples.')
    else:
        synthesis_size = 0
        trainloader_batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=trainloader_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    # Ordered loader for embedding calculation
    ordered_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    if args.max_iters_per_epoch is None:
        args.max_iters_per_epoch = len(train_loader)

    args.max_iters_per_epoch = min(args.max_iters_per_epoch, len(train_loader))
    print(f"Number of iters per epoch: {args.max_iters_per_epoch}")
    clip_wrapper = CLIPWrapper(args.model_name,
                               pretrain=args.pretrain,
                               total_steps=args.max_iters_per_epoch * args.num_epochs,
                               data_parallel=args.data_parallel).cuda()
    
    if args.synthesis and not args.online_sim:
        train_dataset.update_embeddings(clip_wrapper, ordered_train_loader, modality="text")
        if args.preprocess_neighbors:
            train_dataset.preprocess_neighbors(k_neighbors=synthesis_size//args.num_clusters)

    # Best accuracy records
    best_val_acc1, best_val_acc50, best_val_acc200 = 0.0, 0.0, 0.0
    time_0 = time.time()
    for epoch in range(args.num_epochs):
        start_time = time.time()
        if epoch in upweight_epochs:
            args.synthesis_prop *= 2
            trainloader_batch_size, synthesis_size = get_batch_split(args.batch_size, args.synthesis_prop)
            # Create new training loader
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=trainloader_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            args.num_clusters = max(int(synthesis_size / args.cluster_size), 1)
            print(f'Using {synthesis_size} categorized and {args.batch_size - synthesis_size} random samples.')

        if epoch == 0:
            accumulate_step = args.log_offset * args.max_iters_per_epoch
            clip_wrapper.save_model(f'clip_epoch{args.log_offset}_ckpt', args.save_dir, epoch=args.log_offset)
            batch_evaluate(accumulate_step, clip_wrapper, validation_loader)
            evaluate_clip(clip_wrapper, validation_loader, step=accumulate_step)

        # Update embeddings
        if args.online_sim:
            train_dataset.update_embeddings(clip_wrapper, ordered_train_loader, modality=args.online_sim)
            if args.preprocess_neighbors:
                train_dataset.preprocess_neighbors(k_neighbors=synthesis_size//args.num_clusters)

        for i, (image_paths, images, tokens, token_len) in enumerate(train_loader):
            actual_step = i + args.max_iters_per_epoch * (epoch + args.log_offset)
            if args.synthesis:
                _, syn_images, syn_tokens, _ = train_dataset.get_synthesis_batch(
                    synthesis_size,
                    num_clusters=args.num_clusters,
                    neighborhood=args.neighborhood)
                if args.batch_size != synthesis_size:
                    images = torch.cat([syn_images, images])
                    tokens = torch.cat([syn_tokens, tokens])
                else:
                    images = syn_images
                    tokens = syn_tokens

                images, tokens = images.cuda(), tokens.cuda()
                train_batch = (images, tokens)

            else:
                train_batch = (images.cuda(), tokens.cuda())
                cluster_pivots, cluster_elements = [], []

            training_loss, train_i2t, train_t2i = clip_wrapper.training_step(train_batch, actual_step)

            if i % 10 == 0:
                end_time = time.time()
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{args.max_iters_per_epoch}]: '
                      f'training loss {training_loss}, i2t {train_i2t}, t2i {train_t2i}, '
                      f'time {end_time - start_time}')
                start_time = end_time

            if i == args.max_iters_per_epoch - 1:
                # Termination
                break

        time_t = time.time()
        print(f"Total GPU time at epoch {epoch}: {time_t - time_0}")
        accumulate_step = args.max_iters_per_epoch * (epoch + args.log_offset + 1)
        batch_evaluate(accumulate_step, clip_wrapper, validation_loader)

        clip_wrapper.save_model('clip_ckpt', args.save_dir)
        _, val_tot_acc1, val_tot_acc50, val_tot_acc200 = evaluate_clip(clip_wrapper,
                                                                       validation_loader,
                                                                       step=accumulate_step)
        if val_tot_acc1 > best_val_acc1:
            best_val_acc1 = val_tot_acc1
            clip_wrapper.save_model(f'clip_acc1_ckpt', args.save_dir,
                epoch=epoch + 1 + args.log_offset,
                val_tot_acc1=val_tot_acc1.tolist(),
                val_tot_acc50=val_tot_acc50.tolist(),
                val_tot_acc200=val_tot_acc200.tolist())

        if val_tot_acc50 > best_val_acc50:
            best_val_acc50 = val_tot_acc50
            clip_wrapper.save_model(f'clip_acc50_ckpt', args.save_dir,
                epoch=epoch + 1 + args.log_offset,
                val_tot_acc1=val_tot_acc1.tolist(),
                val_tot_acc50=val_tot_acc50.tolist(),
                val_tot_acc200=val_tot_acc200.tolist())

        if val_tot_acc200 > best_val_acc200:
            best_val_acc200 = val_tot_acc200
            clip_wrapper.save_model(f'clip_acc200_ckpt', args.save_dir,
                epoch=epoch + 1 + args.log_offset,
                val_tot_acc1=val_tot_acc1.tolist(),
                val_tot_acc50=val_tot_acc50.tolist(),
                val_tot_acc200=val_tot_acc200.tolist())
            
        if (epoch + 1) % args.save_freq == 0:
            clip_wrapper.save_model(f'clip_epoch{epoch + 1 + args.log_offset}_ckpt',
                args.save_dir,
                epoch=epoch + 1 + args.log_offset,
                val_tot_acc1=val_tot_acc1.tolist(),
                val_tot_acc50=val_tot_acc50.tolist(),
                val_tot_acc200=val_tot_acc200.tolist())


if __name__ == '__main__':
    main()
