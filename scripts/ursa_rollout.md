I’ll inspect your existing training launcher scripts first, then create two matching rollout launchers: one interactive bash script and one sbatch script that can run either rollout_wrf_wofs.py or rollout_metrics_wrf_wofs.py with your current config defaults.

Ran terminal command:  cd /home/Zhanxiang.Hua/miles-credit-wofs && find scripts -maxdepth 2 -type f | sort

Ran terminal command:  cd /home/Zhanxiang.Hua/miles-credit-wofs && find . -maxdepth 3 -type f \( -name "*sbatch*.sh" -o -name "*slurm*.sh" -o -name "*interactive*.sh" -o -name "run_*train*.sh" \) | sort

Read [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts)

I found existing rollout helper scripts, so I’m now reading them plus your training launchers to mirror your preferred style and produce the two new launch scripts exactly in that format.

Read [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts/ursa_interactive_forecast.sh)

Read [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts/ursa_wofscast.sh)

Read [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts/run_rollout_wrf_wofs.sh)

Read [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts/run_rollout_metrics_wrf_wofs.sh)

I’m going to add two new launchers in your `scripts` folder: an sbatch-style script and an interactive bash script, each able to run either rollout forecast or rollout metrics via a single `APP` switch.

Ran terminal command:  cd /home/Zhanxiang.Hua/miles-credit-wofs && find . -name SKILL.md -o -name "*SKILL*.md" | head -n 50

I’m creating the two scripts now with your Ursa-style module/conda/NCCL setup, defaulting to DDP and allowing a single switch to choose `rollout` vs `metrics`.

Created [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts/ursa_interactive_rollout.sh)

Created [](file:///home/Zhanxiang.Hua/miles-credit-wofs/scripts/ursa_rollout.sh)

I’ve created both scripts; next I’ll do quick `bash -n` validation and mark them executable so they are ready to run immediately.

Ran terminal command:  cd /home/Zhanxiang.Hua/miles-credit-wofs && bash -n scripts/ursa_interactive_rollout.sh && bash -n scripts/ursa_rollout.sh && chmod +x scripts/ursa_interactive_rollout.sh scripts/ursa_rollout.sh && ls -l scripts/ursa_interactive_rollout.sh scripts/ursa_rollout.sh

Created both scripts in the same Ursa launcher style as your training scripts, and made them executable.

Files added:
- scripts/ursa_interactive_rollout.sh
- scripts/ursa_rollout.sh

What they do:
- Support both apps with one switch:
  - `APP=rollout` runs `applications/rollout_wrf_wofs.py`
  - `APP=metrics` runs `applications/rollout_metrics_wrf_wofs.py`
- Support `MODE=none|ddp|fsdp` (default `ddp`)
- Use your conda/module stack (`credit-wofs`, `cuda/12.8`)
- Use your current config default: config/ursa_wofscast_credit_wrf_latest.yml
- Keep distributed launch via `torchrun` for multi-GPU
- Include optional `MAX_CASES` and metrics `OUTPUT_NAME`

Quick usage:

Interactive bash:
```bash
scripts/ursa_interactive_rollout.sh
scripts/ursa_interactive_rollout.sh metrics ddp 2
scripts/ursa_interactive_rollout.sh rollout none
```

SBATCH:
```bash
sbatch scripts/ursa_rollout.sh
sbatch --export=ALL,APP=metrics,MODE=ddp,NPROC_PER_NODE_OVERRIDE=2 scripts/ursa_rollout.sh
sbatch --export=ALL,APP=rollout,MODE=ddp scripts/ursa_rollout.sh
```

Validation done:
- `bash -n` passed for both scripts
- execute bit set (`chmod +x`) on both scripts

If you want, I can also add two tiny wrapper scripts (`ursa_rollout_forecast.sh` and `ursa_rollout_metrics.sh`) that hardcode `APP` so you can submit/run without any args.

Made changes.