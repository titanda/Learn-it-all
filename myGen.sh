#MODEL_DIR=/data/jenhaochen/delaney/s_e_cp/
#MODEL_DIR=/data/jenhaochen/delaney/s_e_cp_0_0.1/
#MODEL_DIR=/data/jenhaochen/delaney/origin_cp/
MODEL_DIR=/data/jenhaochen/delaney_dc/eFull_17_1_2_noSame_res/eMax1/
#MODEL_DIR=./checkpoints/fconv/

python interactive.py --cpu --path $MODEL_DIR/checkpoint_best.pt $MODEL_DIR

