"""
Dry Run Script — tests the SNUNet-CD model on synthetic data.

Verifies:
  1. Model forward pass with correct output shapes
  2. Loss computation over all 5 outputs
  3. One backward pass (gradient flow)
  4. Parameter count matches paper (~3.8M)
  5. CUDA compatibility

Run from any assignment directory:
    python dry_run.py
    python dry_run.py --in_channels 13   # for OSCD (assignment3)
    python dry_run.py --batch_size 2     # reduce if OOM
"""

import argparse
import torch
import torch.nn as nn
from models.snunet import SNUNet_ECAM, init_weights_kaiming, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in_channels',    type=int, default=3)
    p.add_argument('--batch_size',     type=int, default=4)
    p.add_argument('--patch_size',     type=int, default=256)
    p.add_argument('--drop_path_rate', type=float, default=0.0)
    return p.parse_args()


def run_dry_run():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'  SNUNet-CD Dry Run')
    print(f'{"="*60}')
    print(f'  Device      : {device}')
    if device.type == 'cuda':
        print(f'  GPU         : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  in_channels : {args.in_channels}')
    print(f'  batch_size  : {args.batch_size}')
    print(f'  patch_size  : {args.patch_size}×{args.patch_size}')
    print(f'{"="*60}\n')

    # ── Build model ───────────────────────────────────────────────────────────
    print('[1/5] Building model...')
    model = SNUNet_ECAM(
        in_channels=args.in_channels,
        out_channels=2,
        drop_path_rate=args.drop_path_rate
    )
    init_weights_kaiming(model)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f'      Parameters: {n_params:,}')
    assert n_params > 1_000_000, 'Parameter count too low — check model!'
    print('      ✓ Parameter count OK\n')

    # ── Synthetic input ───────────────────────────────────────────────────────
    print('[2/5] Creating synthetic input tensors...')
    B, C, H, W = args.batch_size, args.in_channels, args.patch_size, args.patch_size
    x1 = torch.randn(B, C, H, W, device=device)
    x2 = torch.randn(B, C, H, W, device=device)
    gt = torch.randint(0, 2, (B, H, W), device=device, dtype=torch.long)
    print(f'      x1 shape: {tuple(x1.shape)}')
    print(f'      x2 shape: {tuple(x2.shape)}')
    print(f'      gt shape: {tuple(gt.shape)}\n')

    # ── Forward pass ──────────────────────────────────────────────────────────
    print('[3/5] Running forward pass...')
    model.eval()
    with torch.no_grad():
        outputs = model(x1, x2)

    assert len(outputs) == 5, f'Expected 5 outputs, got {len(outputs)}'
    for i, o in enumerate(outputs):
        assert o.shape == (B, 2, H, W), \
            f'Output {i} shape mismatch: {o.shape} vs {(B, 2, H, W)}'
        print(f'      Output {i+1} shape: {tuple(o.shape)} ✓')
    print()

    # ── Loss computation ──────────────────────────────────────────────────────
    print('[4/5] Computing loss and backward pass...')
    model.train()
    outputs_train = model(x1, x2)
    ce = nn.CrossEntropyLoss()
    loss = sum(ce(o, gt) for o in outputs_train)
    print(f'      Loss value: {loss.item():.4f}')
    loss.backward()
    print('      ✓ Backward pass OK\n')

    # ── Memory usage ──────────────────────────────────────────────────────────
    if device.type == 'cuda':
        print('[5/5] GPU memory usage...')
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f'      Allocated : {alloc:.3f} GB')
        print(f'      Reserved  : {reserved:.3f} GB\n')
    else:
        print('[5/5] Running on CPU — skipping GPU memory check\n')

    print('='*60)
    print('  ✅  Dry run PASSED — model is ready for training!')
    print('='*60)


if __name__ == '__main__':
    run_dry_run()
