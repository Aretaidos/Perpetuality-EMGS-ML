# GPU Detection Troubleshooting

## Problem
PyTorch reports `CUDA available: False` even though:
- GPUs are visible via `nvidia-smi`
- PyTorch is compiled with CUDA 12.4
- CUDA module (12.8.1) is available

## Root Cause
The CUDA module must be loaded **before** running PyTorch. The module system sets `LD_LIBRARY_PATH` which PyTorch needs to find CUDA libraries.

## Solution
1. **Load CUDA module in scripts**: Both `run_all_models_uva.sh` and `run_all_models_uva_background.sh` now load `cuda/12.8.1`
2. **Preserve environment**: The `compare_models.py` script preserves `LD_LIBRARY_PATH` when spawning subprocesses

## Verification
After loading the CUDA module, verify:
```bash
module load cuda/12.8.1
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Status
- ✅ Scripts updated to load CUDA module
- ✅ Environment preservation added to subprocess calls
- ⚠️ Still testing if PyTorch detects CUDA after module load

## Next Steps
1. Re-run the comparison script with CUDA module loaded
2. If still failing, may need to reinstall PyTorch with matching CUDA version

