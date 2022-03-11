CUDA_VISIBLE_DEVICES=8 OMP_NUM_THREADS=1 python3 launch.py \
--nproc_per_node=1 --master_port 23658 distributed_slomotrain_Dis_S.py
