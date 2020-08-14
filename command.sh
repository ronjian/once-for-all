# python eval_specialized_net.py \
# --path '/dataset/ILSVRC2012' \
# --net flops@595M_top1@80.0_finetune@75 \
# --gpu '1' \
# --batch-size 64 \
# --workers 4 \


horovodrun -np 2 -H 127.0.0.1:2 python train_ofa_net.py
