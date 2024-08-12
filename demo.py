from model.ea_model import EaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template
import argparse
import torch

"""
next steps:
* measure accuracy
* measure eagle time if possible
* figure out how to specify tree / k
* grid search over k
"""

def main(args):
    base_model_path = "meta-llama/Llama-2-7b-chat-hf"
    EAGLE_model_path = "yuhuili/EAGLE-llama2-chat-7B"

    #base_model_path = "meta-llama/Llama-2-70b-chat-hf"
    #EAGLE_model_path = "yuhuili/EAGLE-llama2-chat-70B"


    use_llama_2_chat = True
    use_vicuna = not use_llama_2_chat
    use_base = args.use_base
    
    if use_base:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=EAGLE_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        tokenizer = model.tokenizer
    model.eval()

    import json
    prompts = []
    with open('/home/ray/workspace-project-cade-dev/test-set.jsonl', 'r') as f:
        for l in f.readlines():
            data = json.loads(l)
            prompts.append(data)
    print(f'loaded {len(prompts)} prompts')
    for i, your_message in enumerate(prompts):
        if i >= 20:
            print(f'{i=}, breaking')
            break
        print(f'prompt {i}')
        run(your_message, model, tokenizer, args, use_llama_2_chat, use_vicuna, use_base)

def run(your_message, model, tokenizer, args, use_llama_2_chat, use_vicuna, use_base):
    
    if use_llama_2_chat:
        conv = get_conversation_template("llama-2-chat")  
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " "
    
    if use_vicuna:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    input_ids = tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    
    print(f'input id shape {input_ids.shape}')
    
    import time
    num_runs = args.num_runs
    
    all_dur_s = []
    all_ms_tok = []
    outputs = []
    output_lens = []
    for i in range(num_runs):
        start_time = time.time()
        if use_base:
            output_ids = model.generate(input_ids, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        else:
            if True:
                from model.choices import mc_sim_7b_63
                tree_choices = mc_sim_7b_63
            else:
                #tree_choices = [[0], [0, 0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
                tree_choices = [[0], [1], [3]]
            output_ids = model.eagenerate(
                input_ids,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                tree_choices=tree_choices,
            )
        end_time = time.time()
    
        dur_s = end_time - start_time
        output_len = len(output_ids[0])
    
        if output_len == 0:
            output_len = float('nan')
    
        ms_tok = dur_s * 1000 / output_len
        tok_s = 1000 / ms_tok
        print(f'{i=} {ms_tok=:.2f} {tok_s=:.2f} {output_len=}')
    
        all_dur_s.append(dur_s)
        all_ms_tok.append(ms_tok)
        output_lens.append(output_len)
        output = tokenizer.decode(output_ids[0])
        outputs.append(output)
    
    warmup_iters = 0
    num_non_warmup = len(all_ms_tok[warmup_iters:])
    avg_ms_tok = sum(all_ms_tok[warmup_iters:]) / num_non_warmup
    avg_tok_s = 1000 / avg_ms_tok
    print(f'summary {args.use_base=} {num_non_warmup=} {avg_ms_tok=:.2f} {avg_tok_s=:.2f}')

if __name__ == '__main__':
    # Create the argparse object
    parser = argparse.ArgumentParser(description='Script arguments')
    
    # Adding arguments
    parser.add_argument('--use-base', action='store_true', help='Use base setting')
    parser.add_argument('--temperature', type=float, required=True, help='Set the temperature')
    parser.add_argument('--max-new-tokens', type=int, required=True, help='Maximum number of new tokens')
    parser.add_argument('--num-runs', type=int, required=True, help='Number of runs')

    # Parsing the arguments
    args = parser.parse_args()

    print(f'running with {args=}')
    main(args)
