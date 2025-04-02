import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tvm
from tvm import relay

# Load Mistral-7B (example)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Create sample input
input_ids = torch.randint(0, 1000, (1, 128))  # (batch_size, seq_len)

# Convert to TVM Relay
input_name = "input_ids"
shape_dict = {input_name: input_ids.shape}
mod, params = relay.frontend.from_pytorch(model, [input_ids], shape_dict)


# Apply quantization (INT8 example)
with tvm.transform.PassContext(opt_level=3):
    quantized_mod = relay.transform.quantize.quantize(
        mod, params
    )

# Target the Alveo U250 via Vitis AI
target = tvm.target.Target(
    "vitis-ai -dpu=DPUCVDX8H -export_runtime_module=model.so",
    host="llvm -mtriple=aarch64-linux-gnu",
)

# Build the model
with tvm.transform.PassContext(opt_level=3, config={"relay.ext.vitis_ai.options.target": "DPUCVDX8H"}):
    lib = relay.build(quantized_mod, target=target, params=params)



# Load the compiled model
from tvm.contrib import vitis_ai

ctx = tvm.cpu()
runtime = vitis_ai.Runtime()
runtime.set_context()
module = runtime.create(lib, ctx)

# Run inference
input_data = tvm.nd.array(input_ids.numpy(), ctx)
output = module.run(input_data)