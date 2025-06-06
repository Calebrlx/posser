import onnx

MODEL_PATH = "bodypix_mobilenet_float_075_224.onnx"

# Load model
model = onnx.load(MODEL_PATH)
graph = model.graph

# Print input details
print("🔹 Inputs:")
for input_tensor in graph.input:
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.HasField("dim_value") else "?")
    print(f" - {input_tensor.name}: shape={shape}")

# Print output details
print("\n🔹 Outputs:")
for output_tensor in graph.output:
    shape = []
    for dim in output_tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.HasField("dim_value") else "?")
    print(f" - {output_tensor.name}: shape={shape}")

# Print all initializers (weights/constants)
print(f"\n🔹 Number of initializers: {len(graph.initializer)}")

# List unique class IDs used in output (optional, need to infer from data if not labeled)
print("\n🔹 Note: Class IDs (e.g., 1=hair, 15=hand) are NOT included in the model directly unless you have metadata.")