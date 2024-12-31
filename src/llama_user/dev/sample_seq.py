import asyncio
import os
from typing import List, Optional

from llamafactory.chat import BaseEngine
from llamafactory.chat.hf_engine import HuggingfaceEngine
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from llamafactory.model import load_tokenizer, load_model
from llamafactory.extras.misc import get_logits_processor
from llamafactory.hparams import ModelArguments, DataArguments, get_infer_args
from llamafactory.data import get_template_and_fix_tokenizer



def sample_sequences(
        text: str,
        model_name_or_path: str,
        template_name: Optional[str] = None,
        num_sequences: int = 3,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
) -> List[str]:
    args = {
        "model_name_or_path": model_name_or_path,
        "template": template_name,
        # Generation arguments
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
    messages = []
    messages.append({"role": "user", "content": text})
    res_list = asyncio.run(engine.chat(messages,
                                       num_return_sequences=num_sequences))
    for response in res_list:
        print(response)


# Example usage
if __name__ == "__main__":
    # Example text
    input_text = "What is common in these texts?"
    msg = [
        "I think I look OK...everybody seems to think I look like ten pounds of shit in a five pound bag.",
        "Nit heroine but a few years ago I had surgery and the pain needs while I was still in the hospital were opiates, as were the pills they gave me when I went home. About a year later I had my wisdom teeth out and the pain killers then were also some kind of opiate. Best. Drug. Ever. Those two instances turned me around on drugs, I used to think everything should be legal, after that experience I totally get how people get addicted.",
        "My ex-wife told me about a guy in her office named Richard Schlichenhardt. Everybody called him Dick."
    ]
    for e in msg:
        input_text += f"<text>{e}</text>\n"

    model_name_or_path = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
    # Generate samples
    samples = sample_sequences(
        text=input_text,
        model_name_or_path=model_name_or_path,  # Replace with your model path
        template_name="llama3",
        num_sequences=10,
        max_length=15,
        temperature=0.7,
        top_p=0.9
    )

    # Print generated sequences
    print("Input text:", input_text)
    print("\nGenerated sequences:")
    for i, sequence in enumerate(samples, 1):
        print(f"\nSequence {i}:")
        print(sequence)