#MODEL_DIR=/data/jenhaochen/delaney/s_e_cp/
MODEL_DIR=/home/arucuid/explain/fairseq/checkpoints/fconv/

python interactive.py --cpu --path $MODEL_DIR/checkpoint_last.pt $MODEL_DIR

