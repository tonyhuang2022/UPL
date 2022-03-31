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
CLASS_EQULE=$7  # CLASS_EQULE True of False


DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLASS_EQULE}_CONF_THRESHOLD_${CONF_THRESHOLD}_RN50_temp/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

python upl_test.py \
--root ${DATA} \
--seed 1 \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
TRAINER.UPLTrainer.N_CTX ${NCTX} \
TRAINER.UPLTrainer.CSC ${CSC} \
TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.CLASS_EQULE ${CLASS_EQULE}



