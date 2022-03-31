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
TAG=$8 # log tag (multiple_models_random_init or rn50_random_init)


for SEED in {1..16}
do
    DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLASS_EQULE}_${CONF_THRESHOLD}_${TAG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python upl_train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.UPLTrainer.N_CTX ${NCTX} \
        TRAINER.UPLTrainer.CSC ${CSC} \
        TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.CLASS_EQULE ${CLASS_EQULE} 
    fi
done


# #!/bin/bash

# cd ..

# # custom config
# DATA=./data
# TRAINER=ZeroshotCLIP
# DATASET=$1
# CFG=$2  # rn50, rn101, vit_b32 or vit_b16

# python sstrain.py \
# --root ${DATA} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/HHTrainer/${CFG}.yaml \
# --output-dir output/${TRAINER}/${CFG}/zeroshot/${DATASET} \
# --eval-only