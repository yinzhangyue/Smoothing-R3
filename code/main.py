import sys
import subprocess


def main():
    # Retriever
    subprocess.call(
        " CUDA_VISIBLE_DEVICES=0 python train.py \
        --task RE \
        --lr 5e-6 \
        --batch-size 1\
        --accumulation-steps 16\
        --epoch 16 \
        --seed 41 \
        --re-model Roberta",
        shell=True,
    )
    # Reader
    # subprocess.call(
    #     " CUDA_VISIBLE_DEVICES=0 python train.py \
    #         --task QA \
    #         --lr 2e-6 \
    #         --batch-size 8\
    #         --accumulation-steps 2\
    #         --epoch 16\
    #         --seed 41\
    #         --qa-model Deberta",
    #     shell=True,
    # )


if __name__ == "__main__":
    main()
