import argparse
from xuyang.utils import reset_args

class PromptTuringArgs:
    def __init__(self) -> None:
        # default
        self.device = 'cuda:0'
        self.device_idx = '7'

        # llm model parameters util도 변경할 것
        self.llm_name = 'qwen2.5-1.5b'  # gpt2, llama-7b, qwen2.5-1.5b, llama-3.2-1b
        if self.llm_name == 'tiny_llama-1.1b':
            self.model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.max_length = 2048
        elif self.llm_name == 'qwen2.5-1.5b':
            self.model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
            self.tokenizer_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
            self.max_length = 32768
        elif self.llm_name == 'llama-3.2-1b':
            self.model_name_or_path = "meta-llama/Llama-3.2-1B"
            self.tokenizer_name_or_path = "meta-llama/Llama-3.2-1B"
            self.max_length = 58180
        elif self.llm_name == 'gpt2':
            self.model_name_or_path = "openai-community/gpt2"
            self.tokenizer_name_or_path = "openai-community/gpt2"
            self.max_length = 1024

        # peft_config
        self.peft_type = 'CAUSAL_LM'
        self.task_type = 'TEXT'
        self.num_virtual_tokens = 50
        self.prompt_tuning_init_text = "please generate query for this document"
        self.peft_model_id = f"{self.llm_name}_{self.peft_type}_{self.task_type}"
        self.prompt_num = 2

        # dataset parameters
        self.dataset_name = "law"
        self.train_data = "xuyang/data/law/prompt_tuning_train_text.csv"
        self.eval_data = "xuyang/data/law/prompt_tuning_test_text.csv"
        self.test_data = "xuyang/data/law/prompt_tuning_test_text.csv"
        self.few_shot_num = 50
        self.fixed_prompt = True
        self.output_log_path = 'train_log'

        # prompt parameters
        self.lr = 3e-2
        self.num_epochs = 20
        self.batch_size = 1
        self.eval_batch_size = 2

        # experiments parameters
        self.checkpoint_name = f"{self.dataset_name}_{self.model_name_or_path}_{self.peft_type}_{self.task_type}_v1.pt".replace("/", "_")
        
        # log file
        self.experiment_dir = 'llm_models'
        self.experiment_description = 'test_pointwise_v1_{}_{}_{}_{}_{}'.format(self.dataset_name, self.peft_model_id, self.num_virtual_tokens, self.few_shot_num, self.prompt_num)


if __name__ == "__main__":
    base_args = PromptTuringArgs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_virtual_tokens", type=int, help="num virtual tokens for prompt")
    parser.add_argument("--llm_name", type=str, help="model name")
    parser.add_argument("--device_idx", type=str, help="device id")
    parser.add_argument("--prompt_num", type=int, help="prompt number")
    parser.add_argument("--num_epochs", type=int, help="epoch number", default=10)
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--train_data", type=str, help="train data path")
    parser.add_argument("--eval_data", type=str, help="eval data path")
    parser.add_argument("--test_data", type=str, help="test data path")
    parser.add_argument("few_shot_num", type=int, help="few shot setting")
    args = parser.parse_args(namespace=base_args)
    args = reset_args(args)
    print(args)