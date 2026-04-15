# GVHMR

```bash
cd GVHMR
python tools/train.py exp=gvhmr/mixed/mixed
```

```bash
cd GVHMR
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
```

```bash
cd GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```

```bash
cd GVHMR
python tools/demo/demo.py --video=/path/to/phone_video.mp4 --output_root outputs/phone_video
```

```bash
cd GVHMR
python tools/demo/demo.py --video=/path/to/phone_video.mp4 --output_root outputs/phone_video -s
```

```bash
cd GVHMR
python tools/demo/demo_folder.py -f /path/to/phone_videos -d outputs/phone_batch
```

# GMR

```bash
cd GMR
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file ../GVHMR/outputs/demo/tennis/hmr4d_results.pt \
  --robot unitree_g1 \
  --save_path output_pkls/unitree_g1/tennis.pkl \
  --record_video
```

```bash
cd GMR
python scripts/batch_gmr_pkl_to_csv.py --folder output_pkls/unitree_g1
```

# whole_body_tracking

```bash
cd whole_body_tracking
python scripts/csv_to_npz.py \
  --input_file ../GMR/output_pkls/unitree_g1/csv/tennis.csv \
  --input_fps 30 \
  --output_name tennis \
  --headless
```

```bash
cd whole_body_tracking
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/tennis
```

```bash
cd whole_body_tracking
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/tennis:v0 \
  --headless \
  --logger wandb \
  --log_project_name G1-Tracking \
  --run_name tennis
```

```bash
cd whole_body_tracking
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```
