# Function-calling & JSON-mode Evaluation
A framework for evaluating function calls and json output by LLMs.

This script evaluates the performance of a language model on a function calling and JSON output tasks. It preprocesses prompts, runs model completions, parses the function calls/json objects in the completions, validates the function calls/json objects, and calculates the pass rate.

## Usage

1. Clone the repository or copy the script to your local machine.
```bash
git clone https://github.com/your-repo/function-calling-eval.git
cd function-calling-eval/tool_eval
```

2. Install the required dependencies:
```bash
pip -r requirements.txt
```

### Arguments

- `--model_path`: Path to the model folder (required).
- `--chat_template`: Chat template for prompt formatting (default: `"chatml"`).
- `--num_fewshot`: Option to subset the evaluation dataset (default: `None`).
- `--dataset_path`: Path to the Hugging Face dataset (default: function-calling: `"NousResearch/func-calling-eval"` & json-mode: `"NousResearch/json-mode-eval"`).
- `--load_in_4bit`: Option to load the model in 4-bit mode with `bitsandbytes` (default: `"False"`).
- `--dpo`: Option to save the dataset for DPO (default: `"False"`).

## Example

### Function-calling
```bash
python evaluator.py --model_path /path/to/model --chat_template chatml --dataset_path dataset/path --load_in_4bit True --dpo False
```
#### Output

The script generates the following outputs:
- `function_calling_eval_results.json`: A JSON file containing the function-calling evaluation results, including prompts, completions, model outputs, and pass/fail status.
- `function_calling_dpo_pairs.json` (if `--dpo` is set to `"True"`): A JSON file containing the DPO dataset for function-calling consisting of system messages, questions, chosen completions, and rejected completions.

### JSON-mode
```bash
python evaluator_json_mode.py --model_path /path/to/model --load_in_4bit True --dpo False
```
#### Output

The script generates the following outputs:
- `json_mode_eval_results.json`: A JSON file containing the json-mode evaluation results, including prompts, completions, model outputs, and pass/fail status.
- `json_mode_dpo_pairs.json` (if `--dpo` is set to `"True"`): A JSON file containing the DPO dataset for json-mode consisting of system messages, questions, chosen completions, and rejected completions.
