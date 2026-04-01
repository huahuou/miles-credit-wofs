
module load cuda/12.6
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs
torchrun --standalone --nnodes 1 --nproc-per-node=1 applications/train_wrf_wofs_multi.py -c /home/zhanxiang.hua/miles-credit-wofs/config/wofs_credit_wrf_t0_multi_train_date_range_example.yml
python scripts/check_nan_batch.py --config /home/zhanxiang.hua/miles-credit-wofs/config/wofscast_credit_wrf_0331.yml --indices 33178 24287 21088 25214

python remove_nonstandard_zarrs.py --base-dir /work2/zhanxianghua/wofs_preprocess_to_credit_0327/cases/ --remove-nan