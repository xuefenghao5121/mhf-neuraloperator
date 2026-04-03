#!/usr/bin/env python3
"""
Quick test to verify the experiment setup
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/home/huawei/.openclaw/workspace/neuraloperator")

import torch
from neuralop.models import FNO

print("="*60)
print("Quick Test - Verify Setup")
print("="*60)

# Test 1: Import FNO
print("\n1. Testing FNO import...")
try:
    model = FNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=2,
    )
    print(f"   ✅ FNO created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Forward pass
print("\n2. Testing forward pass...")
try:
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    print(f"   ✅ Forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Check available models
print("\n3. Checking available models in neuralop...")
try:
    import neuralop.models as nm
    available_models = [name for name in dir(nm) if not name.startswith('_') and name[0].isupper()]
    print(f"   Available models: {available_models}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Import MHF-FNO
print("\n4. Testing MHF-FNO import...")
try:
    from models.fno_mhf import MHFNO
    print(f"   ✅ MHF-FNO imported successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 5: Create MHF-FNO
print("\n5. Testing MHF-FNO creation...")
try:
    mhf_model = MHFNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=2,
        mhf_rank=4,
        mhf_resolutions=[4, 8, 16],
    )
    print(f"   ✅ MHF-FNO created successfully")
    print(f"   Parameters (before decompose): {sum(p.numel() for p in mhf_model.parameters()):,}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Decompose
print("\n6. Testing MHF decomposition...")
try:
    mhf_model.decompose()
    stats = mhf_model.get_compression_stats()
    print(f"   ✅ Decomposition successful")
    print(f"   Compression factor: {stats['overall_compression_factor']:.2f}x")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Forward pass with decomposed model
print("\n7. Testing forward pass with decomposed model...")
try:
    y_mhf = mhf_model(x)
    print(f"   ✅ Forward pass successful")
    print(f"   Output shape: {y_mhf.shape}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
