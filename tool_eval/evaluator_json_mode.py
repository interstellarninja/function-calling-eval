import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from validator import validate_json_completion, validate_json_data

from utils import (
    eval_logger,
    calculate_pass_rate,
    get_assistant_message,
    get_chat_template,
)

class ModelEvaluator:
    def __init__(self, model_path, chat_template, load_in_4bit, flash_attn, dpo):
        self.bnb_config = None

        if load_in_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Prepare model loading arguments
        model_args = {
            "trust_remote_code": True,
            "return_dict": True,
            "quantization_config": self.bnb_config,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        # Conditionally add attn_implementation if flash_attn is True
        if flash_attn:
            model_args["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_args
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = get_chat_template(chat_template)

        self.eval_results = []
        if dpo:
            self.dpo_results = []
        
        eval_logger.info(self.model.config)
        eval_logger.info(self.model.generation_config)
        eval_logger.info(self.model.parameters)
        eval_logger.info(self.tokenizer.chat_template)
        eval_logger.info(self.tokenizer.special_tokens_map)

    def evaluate_model(self, eval_dataset, chat_template, num_fewshot):

        for sample in tqdm(eval_dataset, desc="processing samples", unit="sample"):  
            prompt = sample['prompt']
            
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors='pt'
            )

            tokens = self.model.generate(
                inputs.to(self.model.device),
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )

            completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False)
            eval_logger.info(f"model completion with eval prompt:\n{completion}")

            assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)

            sample['model_completion'] = ""
            sample['result'] = "failed"

            if assistant_message is not None:
                sample['model_completion'] = assistant_message
                validation, json_object = validate_json_data(assistant_message, json.loads(sample['schema']))
                if validation:
                    result = validate_json_completion(json_object, json.loads(sample['completion']))
                    if result == "failed":
                        eval_logger.info("Json completion validation failed")
                    else:
                        sample['result'] = "passed"
                        sample['model_completion'] = json_object
                        eval_logger.info(f"all validations passed")
                        eval_logger.info(f"parsed json object:\n{json.dumps(json_object, indent=2)}")
            
            if sample['result'] == "failed":
                eval_logger.info("all validations failed")
                eval_logger.info(f"failed assistant message:\n{sample['model_completion']}")

                if hasattr(self, 'dpo_results'):
                    self.dpo_results.append({
                        "system": prompt[0]["content"],
                        "question": prompt[1]['content'],
                        "chosen": sample['completion'],
                        "rejected": assistant_message
                    })

            self.eval_results.append(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on fireworks-ai dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to subset eval dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="Huggingface dataset path")
    parser.add_argument("--flash_attn", type=bool, default=False, help="weather to use flash attention; requires installing flash-attn")
    parser.add_argument("--load_in_4bit", type=str, default=False, help="Option to load in 4bit with bitsandbytes")
    parser.add_argument("--dpo", type=str, default=False, help="Option to save dpo dataset")
    args = parser.parse_args()

    # load eval dataset
    if args.dataset_path:
        eval_dataset = load_dataset(args.dataset_path)["train"]
    else:
        eval_dataset = load_dataset("NousResearch/json-mode-eval")['train']

    # Create model evaluator instance
    model_evaluator = ModelEvaluator(args.model_path, args.chat_template, args.load_in_4bit, args.flash_attn, args.dpo)

    # Run the model evaluator
    model_evaluator.evaluate_model(eval_dataset, args.chat_template, args.num_fewshot)

    # Calculate and print pass rate
    pass_rate = calculate_pass_rate(model_evaluator.eval_results)
    eval_logger.info(f"json-mode eval (pass@1): {pass_rate}")

    results_path = './json_mode_eval_results.json'
    with open(results_path, 'w') as file:
        json.dump(model_evaluator.eval_results, file)

    if args.dpo:
        dpo_path = './json_mode_dpo_pairs.json'
        with open(dpo_path, 'w') as file:
            json.dump(model_evaluator.dpo_results, file)
