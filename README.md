# SimCLIP
An empirical study of CLIP fine-tuning with similarity clusters.

## Preparing Data

SimCLIP experiment is originally trained on COCO dataset. After following the instructions on the website to download the captions and images, reformat it into a `.tsv` file with the following columns,

```
image    caption1        caption2 ...
<image>  <description1>  <description2>
```
We include two placeholder files in our repo under `datasets/`, and you will have to replace them with your desired data.

## Training

You can run the main script for pre-training or fine-tuning CLIP models with or without similarity clustering. We use `wandb` for logging purpose, so you may need to export your wandb api key with

```
export WANDB_API_KEY='your-api-key-here'
```
 
We also provide several examples of command line flags here.

- SimCLIP `k=16`, `p=1.0` and `s=1`.

```
python main.py \
    --data-parallel \
    --batch-size 1024 \
    --val-batch-size 1024 \
    --train_tsv=./datasets/coco_train.tsv \
    --val_tsv=./datasets/coco_val.tsv \
    --train-image-dir=/path/to/train/directory/ \
    --val-image-dir=/path/to/test/directory/ \
    --num-epochs 20 \
    --save-dir=/path/to/save/directory/ \
    --model-name=openai/clip-vit-base-patch32 \
    --exp-name=clip_exp \
    --synthesis \
    --online-sim=text \
    --max-iters-per-epoch 116 \
    --num-clusters=64 \
    --synthesis-prop=1.0 \
    --seed 1
```

- SimCLIP `k=16`, `p=1.0` and `s=1` with warmup.

```
python main.py \
    --data-parallel \
    --batch-size 1024 \
    --val-batch-size 1024 \
    --train_tsv=./datasets/coco_train.tsv \
    --val_tsv=./datasets/coco_val.tsv \
    --train-image-dir=/path/to/train/directory/ \
    --val-image-dir=/path/to/test/directory/ \
    --num-epochs 20 \
    --save-dir=/path/to/save/directory/ \
    --model-name=openai/clip-vit-base-patch32 \
    --exp-name=clip_exp \
    --synthesis \
    --warmup-prop\
    --online-sim=text \
    --max-iters-per-epoch 116 \
    --cluster-size 16 \
    --synthesis-prop 1/512 \
    --seed 1
```

## Evaluation
For evaluations, please follow the standard evaluations in [clip_benchmark](https://github.com/LAION-AI/CLIP_benchmark)

