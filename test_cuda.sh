#!/bin/bash
echo "=== Test 1: Without module or CUDA_VISIBLE_DEVICES ==="
/u/usz7pc/.conda/envs/neuromotor/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo -e "\n=== Test 2: With CUDA_VISIBLE_DEVICES=0 ==="
CUDA_VISIBLE_DEVICES=0 /u/usz7pc/.conda/envs/neuromotor/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo -e "\n=== Test 3: With cuda module loaded ==="
module load cuda/12.8.1 2>/dev/null
/u/usz7pc/.conda/envs/neuromotor/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo -e "\n=== Test 4: With cuda module + CUDA_VISIBLE_DEVICES=0 ==="
CUDA_VISIBLE_DEVICES=0 /u/usz7pc/.conda/envs/neuromotor/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
