import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

if __name__ == "__main__":
    model_path = "meta-llama/Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    for n, p in model.named_parameters():
        print(f"Param {n} on device {p.device}")
