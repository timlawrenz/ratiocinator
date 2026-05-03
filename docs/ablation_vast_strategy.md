# Single-Host Quad-GPU Strategy for Rigorous Ablations

When comparing hyperparameters or model architectures (such as TREAD routing and Muon optimization), it is crucial to eliminate hardware variances. If arms are run on different machines, the following "Silent Bottlenecks" can falsify the iterations-per-second (`s/it`) or convergence metrics:

1. **PCIe Bandwidth:** A GPU in a PCIe Gen4 x16 slot (25 GB/s) versus a 1x riser (1 GB/s) will drastically alter data transfer times and gradient accumulation speed.
2. **CPU Contention:** Modern `webdataset` loading heavily utilizes the CPU to untar and decode files on-the-fly. Different host CPUs will bottleneck the dataloader differently.
3. **Storage Medium:** NVMe versus SATA SSDs impact sequential read speeds.

## The Bulletproof Solution
We utilize a single `4x RTX 4090` machine via `vast.ai`. 
All four arms run concurrently on the exact same physical motherboard. They share the same CPU, PCIe topology, NVMe drive, and PyTorch/CUDA environment. 

### Why this data is unassailable:
1. **Identical Iteration Timing:** With 4 models isolated to 4 GPUs (`CUDA_VISIBLE_DEVICES=0..3`), any variance in computation time is strictly algorithmic.
2. **RAM Dataset Caching:** The 168GB training dataset fits entirely into the host's 256GB RAM. The Linux page cache intercepts the disk reads after the first epoch, eliminating I/O bottlenecks.
3. **Symmetric Contention:** While the models contend for host resources, they contend equally. 
