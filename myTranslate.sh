MODEL_DIR=/home/arucuid/explain/fairseq/checkpoints_bak/fconv/
python interactive.py --path $MODEL_DIR/checkpoint_best.pt $MODEL_DIR --beam 5 --source-lang de --target-lang en

