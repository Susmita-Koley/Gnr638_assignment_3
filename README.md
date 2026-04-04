# Assignment 3 — Ultimate SNUNet Architecture Testing (RAdam + Lovász)

The finalized culmination framework testing ultimate bounds for aerial anomaly generation targeting the `Siam-NestedUNet` architecture.

## 1. Architectural Configuration
The architecture undergoes standard `n1=16` normalization to execute `~2.56M` / `~3.01M` parameters stably on personal Nvidia GPUs cleanly via 256px configurations.

## 2. Experimental Setup: Global Topological Bounds
- **Dataset:** LEVIR-CD (150 Sample Subset via `seed=777`)
- **VHR Handling:** Processed entirely offline via uniform gridding resulting in an absolute maximal `2400` distinct static training sub-patches. 
- **Optimizer:** Advanced `RAdam` (Rectified Adam to stabilize variance).
- **Learning Rate Schedule:** Super-convergence utilizing `OneCycleLR` (Warmup + Cosine Annealing integrated natively per-batch).
- **Criterion:** Ultimate **Lovász-Softmax + BCE Loss**. The Jaccard/IoU proxy is mapped dynamically to continuous space for globally optimal spatial mapping.
- **Validation Additions:** Introduced Multi-scale evaluation constraints (Validation dynamically scales images at `0.75x` and `1.25x` dynamically smoothing probability thresholds).
- **Network Alterations:** Stochastic Depth (`DropPath` mechanisms) implemented within bottleneck matrices physically preventing co-adaptations.

## 3. Environment Execution Protocol

### Step 1: Native Lovász Training Matrix
```powershell
python train.py --data_root data --epochs 10 --batch_size 8 --max_train 2400 --max_val 1024 --seed 777
```

### Step 2: Official Configuration Testing
```powershell
python train_official.py --data_root data --epochs 10 --batch_size 8 --max_train 2400 --max_val 1024 --seed 777
```

### Step 3: Cross-Model Numerical Extraction 
```powershell
python compare_official.py --data_root data --our_ckpt checkpoints/best_model.pt --official_ckpt checkpoints/official_best_model.pt --official_path ../officialSNUNet --seed 777
```
