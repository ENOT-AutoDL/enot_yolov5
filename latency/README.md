### All latencies here were computed with the following setup:

* Batch size - 32
* Image size - 640x640
* Workstation:
  * CPU - AMD Ryzen Threadripper 1950X 16-Core
  * GPU - 4x NVIDIA GeForce RTX 2080 Ti
* Framework - PyTorch 1.9 (conda install pytorch==1.9 torchvision torchaudio cudatoolkit=10.2 -c pytorch)
* CUDA - 10.2
* Data type - float16 (half precision)
* Execution properties:
  * execution was done on a single NVIDIA 2080 Ti
  * memory format is contiguous
  * CUDNN is enabled
  * CUDNN benchmark flag is set to True
  * deterministic execution is disabled
  * model was set to eval() mode
  * batch norm layers were fused with convolutions
