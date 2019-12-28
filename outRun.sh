#mkdir -p "/data/jenhaochen/delaney/eMax$1"
mkdir -p "/data/jenhaochen/delaney/eMax_origin$1"

#nohup python3 train.py --dataset hepth 2>&1 &

#origin
#python train.py data-bin/iwslt14.tokenized.de-en     --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000     --arch fconv_iwslt_de_en --distributed-world-size 1  --save-dir  checkpoints/fconv 2>&1 >out

#tune parameter
#python train.py data-bin/iwslt14.tokenized.de-en --seed $1 --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --distributed-world-size 1 --save-interval=1000 --max-epoch=10000 --save-dir  checkpoints/fconv 2>&1 >out
#python train.py data-bin/iwslt14.tokenized.de-en --seed $1 --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --distributed-world-size 1 --save-interval=1000 --max-epoch=10000 --save-dir  checkpoints/fconv


python train.py data-bin/iwslt14.tokenized.de-en --seed $1 --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --distributed-world-size 1 --save-interval=1 --max-epoch=10000 --save-dir /data/jenhaochen/delaney/eMax$1 2>&1 >/data/jenhaochen/delaney/eMax$1/out$1
#python train.py data-bin/iwslt14.tokenized.de-en --seed $1 --clip-norm 0.1 --dropout 0.2 --arch fconv_iwslt_de_en --distributed-world-size 1 --save-interval=1 --max-epoch=10000 --save-dir /data/jenhaochen/delaney/eMax_origin$1 2>&1 >/data/jenhaochen/delaney/eMax_origin$1/out$1



#python train.py data-bin/iwslt14.tokenized.de-en --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --distributed-world-size 1 --save-interval=1 --max-epoch=10000 --save-dir  /data/jenhaochen/delaney/e_t_cp 2>&1 >/data/jenhaochen/delaney/e_t_cp/out

#reset_lr_scheduler
#python train.py data-bin/iwslt14.tokenized.de-en --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --distributed-world-size 1 --save-interval=50 --max-epoch=1000 --reset-lr-scheduler --save-dir  checkpoints/fconv 2>&1 >out

#adam_meteor
#python train.py data-bin/iwslt14.tokenized.de-en     --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000     --arch fconv_iwslt_de_en --distributed-world-size 1  --optimizer='adam' --lr-scheduler='fixed' --save-interval=100 --max-epoch=1000 --save-dir  checkpoints/fconv 2>&1 >out
