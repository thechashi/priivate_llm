import json
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login
import click
import toml

# Function to print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# Function to load and preprocess training data
def load_and_preprocess_training_data(training_data_path, tokenizer):
    with open(training_data_path) as f:
        train_data = json.load(f)

    def preprocess_training_data(row):
        try:
            training_data = []
            response_text = row['response']
            prompt_text = row['prompt']
            full_text = prompt_text + '\n' + response_text + ' ' + tokenizer.eos_token
            tokens = tokenizer(full_text, return_tensors='pt')
            if tokens.input_ids.shape[1] < 2000:
                training_data.append({'text': full_text})
            return training_data
        except:
            return []

    training_data = [data for row in train_data for data in preprocess_training_data(row)]
    print(training_data[0])
    df = pd.DataFrame(training_data)
    dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())
    hg_dataset = Dataset(pa.Table.from_pandas(df))

    dataset = DatasetDict({'train' : hg_dataset})
    data = dataset
    data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)
    return data


# Function to train the model
def train_model(model, data, training_args, tokenizer):
    trainer = Trainer(
        model=model,
        train_dataset=data["train"].shuffle(),
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()

@click.command()
@click.option('--config-file', default='src/configs/summarizer.toml', help='Path to the TOML configuration file')
def main(config_file):
    # Load configuration from TOML file
    config = toml.load(config_file)

    # Initialize the model and tokenizer
    model_id = config['model']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, torch_dtype=torch.float16)

    # Prepare the model for k-bit trainin
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoraConfig and get_peft_model
    lora_config = LoraConfig(**config['lora_config'])
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Load and preprocess training data
    training_data_path = config['model']['training_data_path']
    data = load_and_preprocess_training_data(training_data_path, tokenizer)

    # Initialize Trainer and start training
    training_args = TrainingArguments(**config['training_arguments'])
    train_model(model, data, training_args, tokenizer)
    
    login()
    model.push_to_hub(config['meta']['adapter_path'])

if __name__ == "__main__":
    main()
