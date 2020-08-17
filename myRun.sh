python train.py data-bin/iwslt14.tokenized.de-en --seed $1 --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --save-interval=1 --max-epoch=5 --distributed-world-size 1 --save-dir checkpoints/fconv 2>&1 >out

