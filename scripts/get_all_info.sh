
DATASET=$1


bash get_info.sh ${DATASET} anay_rn50 end 16 -1 False
bash get_info.sh ${DATASET} anay_rn50x4 end 16 -1 False
bash get_info.sh ${DATASET} anay_rn50x16 end 16 -1 False
bash get_info.sh ${DATASET} anay_rn50x64 end 16 -1 False
bash get_info.sh ${DATASET} anay_rn101 end 16 -1 False
bash get_info.sh ${DATASET} anay_ViT_B_16 end 16 -1 False
bash get_info.sh ${DATASET} anay_ViT_B_32 end 16 -1 False
bash get_info.sh ${DATASET} anay_ViT_L_14 end 16 -1 False