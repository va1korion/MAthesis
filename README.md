# MA Thesis repo

To reproduce:
```shell
git clone https://github.com/Xilinx/Vitis-AI.git
cp Dockerfile.vitis_ai VitisA-AI/third_party/tvm
cd VitisA-AI/third_party/tvm
./build.sh vitis_ai bash
cd ../../
./docker_run.sh tvm.vitis_ai
# Now inside docker...
conda activate vitis-ai-tensorflow
source /workspace/setup/alveo/setup.sh DPUCADF8H

```