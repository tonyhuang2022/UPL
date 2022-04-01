#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=UPLTrainer

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)


for SEED in 1
do
    DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_random_init/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

    echo "Run this job and save the output to ${DIR}"
    python get_info.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.UPLTrainer.N_CTX ${NCTX} \
    TRAINER.UPLTrainer.CSC ${CSC} \
    TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
done
