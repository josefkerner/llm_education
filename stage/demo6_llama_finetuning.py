from model.llama.llama_model import LlamaModel
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

from transformers import TrainingArguments

class LammaFinetuner:
    def __init__(self):
        base_model_name = "NousResearch/Llama-2-7b-chat-hf"
        self.llm = LlamaModel(
            cfg={
                "model_name": base_model_name
            }
        )
    def load_training_data(self):
        data_name = "mlabonne/guanaco-llama2-1k"
        training_data = load_dataset(data_name, split="train")
        print(training_data.data)
        return training_data
    def test_finetuning(self, training_data):
        '''
        Will test finetuning on a specific dataset
        :param training_data:
        :return:
        '''
        # Load model
        self.llm.load_model()

        refined_model = "llama-2-7b-granton"  # You can give it your own name

        # LoRA Config
        peft_parameters = LoraConfig(
            # higher values make the approximation more influential in finetuning
            lora_alpha=16,
            # dropout rate for the LoRA layer (0.0 means no dropout)
            lora_dropout=0.1,
            #rank of the decomposition matrix
            r=8,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Training Params
        train_params = TrainingArguments(
            output_dir="./results_modified",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant"
        )
        # Trainer
        fine_tuning = SFTTrainer(
            model=self.llm.model_bf16,
            train_dataset=training_data,
            peft_config=peft_parameters,
            dataset_text_field="text",
            tokenizer=self.llm.tokenizer,
            args=train_params
        )

        # Training
        fine_tuning.train()

        # Save Model
        fine_tuning.model.save_pretrained(refined_model)

if __name__ == "__main__":
    lft = LammaFinetuner()
    training_data = lft.load_training_data()
    #lft.test_finetuning(training_data=training_data)