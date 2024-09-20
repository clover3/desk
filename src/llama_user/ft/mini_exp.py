from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig
from llmtuner import LoraConfig, TrainerCallback, ModelArguments, DataArguments, FinetuningArguments
from llmtuner import run_sft

# Define mock training data
train_data = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare wrote 'Romeo and Juliet'."),
    ("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter.")
]

# Convert data to Dataset format
dataset = Dataset.from_dict({
    "instruction": [item[0] for item in train_data],
    "output": [item[1] for item in train_data]
})

# Define custom small model configuration
model_name = "meta-llama/Llama-2-7b-hf"  # We'll use this as a base for our custom config
original_config = AutoConfig.from_pretrained(model_name)

custom_config = AutoConfig.from_pretrained(model_name)
custom_config.hidden_size = 256  # Reduced from original (usually 4096)
custom_config.intermediate_size = 512  # Reduced from original (usually 11008)
custom_config.num_attention_heads = 8  # Reduced from original (usually 32)
# Keep the number of layers the same
custom_config.num_hidden_layers = original_config.num_hidden_layers

# Initialize tokenizer and model with custom config
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(custom_config)
