from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LlamaChatbot:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.conversation = []

    def generate_response(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})
        input_text = self.tokenizer.apply_chat_template(self.conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_text = response.split("Assistant:")[-1].strip()
        self.conversation.append({"role": "assistant", "content": response_text})
        return response_text

    def chat(self):
        print("LLaMA 3.1 Instruct Chatbot. Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = self.generate_response(user_input)
            print("LLaMA:", response)


if __name__ == "__main__":
    chatbot = LlamaChatbot()
    chatbot.chat()
