"""
Compare our SNUNet-CD implementation vs. the official repo.

Validates:
  1. Parameter count matches official implementation
  2. Forward pass output shapes are identical
  3. (Optional) Numerical output comparison when both use same random weights

Usage:
    # Compare just our model (standalone)
    python compare.py

    # Compare with official model (requires cloning official repo)
    git clone https://github.com/likyoo/Siam-NestedUNet official_repo
    python compare.py --official_path official_repo
"""

import argparse
import sys
import torch
import torch.nn as nn
from models.snunet import SNUNet_ECAM, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--official_path', type=str, default=None,
                   help='Path to cloned official Siam-NestedUNet repo')
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--batch_size',  type=int, default=2)
    p.add_argument('--patch_size',  type=int, default=256)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n' + '='*65)
    print('  SNUNet-CD Implementation Comparison')
    print('='*65)

    # ── Our implementation ────────────────────────────────────────────────────
    print('\n[OUR IMPLEMENTATION]')
    our_model = SNUNet_ECAM(in_channels=args.in_channels, out_channels=2)
    our_model = our_model.to(device).eval()
    our_params = count_parameters(our_model)
    print(f'  Parameters: {our_params:,}')

    B, C, H, W = args.batch_size, args.in_channels, args.patch_size, args.patch_size
    x1 = torch.randn(B, C, H, W, device=device)
    x2 = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        our_outs = our_model(x1, x2)

    print(f'  Output shapes: {[tuple(o.shape) for o in our_outs]}')
    print(f'  Output 5 (ensemble) min/max: '
          f'{our_outs[-1].min().item():.4f} / {our_outs[-1].max().item():.4f}')

    # ── Official implementation (if available) ────────────────────────────────
    if args.official_path is not None:
        print(f'\n[OFFICIAL IMPLEMENTATION] ({args.official_path})')
        try:
            sys.path.insert(0, args.official_path)
            from models.Models import SNUNet_ECAM as OfficialModel
            official_model = OfficialModel(in_channels=args.in_channels,
                                           out_channels=2)
            official_model = official_model.to(device).eval()
            official_params = sum(p.numel() for p in official_model.parameters()
                                  if p.requires_grad)
            print(f'  Parameters: {official_params:,}')

            # Copy our weights to official model for fair comparison
            try:
                official_model.load_state_dict(our_model.state_dict())
                print('  Weights: successfully loaded from our model')

                with torch.no_grad():
                    off_outs = official_model(x1, x2)

                print(f'  Output shapes: {[tuple(o.shape) for o in off_outs]}')

                # Numerical comparison
                print('\n[NUMERICAL COMPARISON]')
                for i, (our_o, off_o) in enumerate(zip(our_outs, off_outs)):
                    diff = (our_o - off_o).abs().max().item()
                    match = '✓ MATCH' if diff < 1e-4 else f'✗ DIFF={diff:.6f}'
                    print(f'  Output {i+1}: {match}')

            except RuntimeError as e:
                print(f'  Could not transfer weights (architectures differ): {e}')
                print('  Running separate forward pass with random weights...')
                with torch.no_grad():
                    off_outs = official_model(x1, x2)
                print(f'  Output shapes: {[tuple(o.shape) for o in off_outs]}')

            # Parameter count comparison
            print('\n[PARAMETER COUNT]')
            print(f'  Our model     : {our_params:,}')
            print(f'  Official model: {official_params:,}')
            diff_pct = abs(our_params - official_params) / official_params * 100
            if diff_pct < 0.1:
                print(f'  Difference    : {diff_pct:.2f}% ✓ MATCH')
            else:
                print(f'  Difference    : {diff_pct:.2f}% ← architecture differs')

        except ImportError as e:
            print(f'  Could not import official model: {e}')
            print('  Make sure to clone: https://github.com/likyoo/Siam-NestedUNet')
    else:
        print('\n[OFFICIAL COMPARISON SKIPPED]')
        print('  To compare with official, run:')
        print('    git clone https://github.com/likyoo/Siam-NestedUNet official_repo')
        print('    python compare.py --official_path official_repo')

    print('\n' + '='*65)
    print(f'  Our implementation: {our_params:,} parameters')
    print(f'  Paper reports     : ~3.8M parameters (RGB 3-channel model)')
    print('='*65)


if __name__ == '__main__':
    main()
