import torch
import numpy as np

def estimate_gpu_memory_usage(model, input_shape, batch_size, dtype=torch.float32):
    """
    Estimate GPU memory usage for a given model and input shape.
    
    :param model: PyTorch model
    :param input_shape: Shape of a single input (excluding batch dimension)
    :param batch_size: Batch size
    :param dtype: Data type of the input (default: torch.float32)
    :return: Estimated memory usage in GB
    """
    def get_model_parameters_number(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_layer_output_sizes(model, input_shape):
        x = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        layer_output_sizes = []
        hooks = []

        def hook(module, input, output):
            layer_output_sizes.append(output.numel())

        for layer in model.modules():
            hooks.append(layer.register_forward_hook(hook))

        model(x)

        for h in hooks:
            h.remove()

        return layer_output_sizes

    model.to('cuda')
    model.eval()

    # Estimate memory for model parameters
    param_memory = get_model_parameters_number(model) * dtype().element_size() / (1024**3)

    # Estimate memory for input
    input_memory = np.prod(input_shape) * batch_size * dtype().element_size() / (1024**3)

    # Estimate memory for output
    dummy_input = torch.rand(1, *input_shape).cuda()
    dummy_output = model(dummy_input)
    output_memory = dummy_output.numel() * batch_size * dtype().element_size() / (1024**3)

    # Estimate memory for intermediate activations
    layer_output_sizes = get_layer_output_sizes(model, input_shape)
    activation_memory = sum(layer_output_sizes) * batch_size * dtype().element_size() / (1024**3)

    # Estimate memory for gradients (assume same as parameters)
    gradient_memory = param_memory

    # Add some buffer for CUDA context and other overheads (e.g., 1 GB)
    buffer_memory = 1

    total_memory = param_memory + input_memory + output_memory + activation_memory + gradient_memory + buffer_memory

    print(f"Estimated GPU memory usage:")
    print(f"Model parameters: {param_memory:.2f} GB")
    print(f"Input: {input_memory:.2f} GB")
    print(f"Output: {output_memory:.2f} GB")
    print(f"Activations: {activation_memory:.2f} GB")
    print(f"Gradients: {gradient_memory:.2f} GB")
    print(f"Buffer: {buffer_memory:.2f} GB")
    print(f"Total estimated memory: {total_memory:.2f} GB")

    return total_memory