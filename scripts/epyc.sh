
module load cuda/12.6
source /home/zhanxiang.hua/miniconda3/etc/profile.d/conda.sh
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs

OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node=1 applications/train_wrf_wofs_multi.py -c /home/zhanxiang.hua/scratch/credit_runs/wofs_wrf_experiment_multi_0401/model.yml


# export CUDA_VISIBLE_DEVICES=1
# python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node=1 applications/train_wrf_wofs_multi.py -c /home/zhanxiang.hua/miles-credit-wofs/config/wofscast_credit_wrf_0331.yml
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node=2 applications/train_wrf_wofs_da.py -c /home/zhanxiang.hua/miles-credit-wofs/config/wofs_credit_wrf_da_increment.yml
 

# python scripts/check_nan_batch.py --config /home/zhanxiang.hua/miles-credit-wofs/config/wofscast_credit_wrf_0331.yml --indices 33178 24287 21088 25214

# python remove_nonstandard_zarrs.py --base-dir /work2/zhanxianghua/wofs_preprocess_to_credit_0327/cases/ --remove-nan

# python applications/rollout_wrf_wofs_da.py config/wofs_credit_wrf_da_increment.yml


# Single figure:
# python applications/plot_rollout_wrf_wofs_da.py /work2/zhanxianghua/da_test/wofs_da_increment_experiment_0406/eval_valid/wofs_20210406_0000_mem01/rollout_da_wofs_20210406_0000_mem01.nc --var qrain --time-index 0 --level-index 2 --out-dir /work2/zhanxianghua/da_test/wofs_da_increment_experiment_0406/verif_plots
# Batch all slices:
# python applications/plot_rollout_wrf_wofs_da.py /path/to/rollout_da_case.nc --var qrain --all
# Batch both variables:
# python applications/plot_rollout_wrf_wofs_da.py /work2/zhanxianghua/da_test/wofs_da_increment_experiment_0406/eval_valid/wofs_20210406_0000_mem01/rollout_da_wofs_20210406_0000_mem01.nc --all --out-dir /work2/zhanxianghua/da_test/wofs_da_increment_experiment_0406/verif_plots
