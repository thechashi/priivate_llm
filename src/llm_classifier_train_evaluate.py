import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from importlib import reload
import click
import math
import gc
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(
    encoding="utf-8",
    format="%(asctime)s : %(levelname)s : %(module)s:%(lineno)d - %(message)s",
    level=logging.DEBUG,
)
MAX_LENGTH = 1500
BATCH_SIZE = 1 
OPTIMIZE_STEP = 1

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model():
    model_id = "bigscience/bloomz-560m"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    from peft import prepare_model_for_kbit_training

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=128,
        #     target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer


def get_dataset(dataset_json):
    # load the dataset
    with open(dataset_json, "r") as f:
        dataset_json = json.load(f)
    for i in range(len(dataset_json)):
        dataset_json[i]['label'] = int(dataset_json[i]['label'])

    # split the dataset into train and test
    df = pd.DataFrame(dataset_json)
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # log the number of samples in train and test
    logging.info(f"Number of samples in train: {len(train)}")
    logging.info(f"Number of samples in test: {len(test)}")

    # log the number of samples with 0 and 1 label in train and test
    logging.info(
        f"Number of samples with 0 label in train: {len(train[train['label'] == 0])} "
    )
    logging.info(
        f"Number of samples with 1 label in train: {len(train[train['label'] == 1])}"
    )
    logging.info(
        f"Number of samples with 0 label in test: {len(test[test['label'] == 0])} "
    )
    logging.info(
        f"Number of samples with 1 label in test: {len(test[test['label'] == 1])}"
    )

    return train, test


"""
Returns the prompt for the given resume and label
"""


def get_prompt(resume, label, tokenizer):
    instruction = """###Instruction: There is a resume below. You are in the admission committee and you are asked to decide whether to admit this applicant. So, read the resume and decide if you want to admit this applicant, type 'Decision: 1' in the box below. If you reject this applicant, then type "Decision: 0" in the box below.\n
###Resume: {}.\n
###Decision: {}""".format(
        resume, label
    )

    if len(tokenizer(instruction)["input_ids"]) > MAX_LENGTH:
        return None

    return instruction


class CustomDataset:
    def __init__(self, df, tokenizer, balanced_sampling=True):
        self.df = df
        self.df_0 = self.df[self.df["label"] == 0]
        self.df_1 = self.df[self.df["label"] == 1]

        self.indexes_0 = [i for i in range(len(self.df_0))]
        self.indexes_1 = [i for i in range(len(self.df_1))]

        self.count_0 = 0
        self.count_1 = 0
        self.tokenizer = tokenizer
        self.balanced_sampling = balanced_sampling

    def __len__(self):
        return len(self.df)

    def get_next_batch(self):
        data = []

        while len(data) <= BATCH_SIZE:
            if self.balanced_sampling:
                if self.count_0 >= len(self.indexes_0):
                    self.count_0 = 0
                    random.shuffle(self.indexes_0)
                if self.count_1 >= len(self.indexes_1):
                    self.count_1 = 0
                    random.shuffle(self.indexes_1)

                if BATCH_SIZE == 1:
                    # Randomly choose between df_0 and df_1 with equal probability
                    if random.random() < 0.5:
                        i = self.indexes_0[self.count_0]
                        self.count_0 += 1
                        label = self.df.iloc[i]["label"]
                        resume = self.df_0.iloc[i]["resume"][:6500]
                    else:
                        i = self.indexes_1[self.count_1]
                        self.count_1 += 1
                        label = self.df.iloc[i]["label"]
                        resume = self.df_1.iloc[i]["resume"][:6500]
                else:
                    # The existing code to select from both df_0 and df_1
                    i = self.indexes_0[self.count_0]
                    self.count_0 += 1
                    label = self.df.iloc[i]["label"]
                    resume = self.df_0.iloc[i]["resume"][:6500]
                    prompt = get_prompt(resume, label, self.tokenizer)
                    if prompt:
                        data.append(prompt)
                    if len(data) <= BATCH_SIZE:
                        break

                    i = self.indexes_1[self.count_1]
                    self.count_1 += 1
                    label = self.df.iloc[i]["label"]
                    resume = self.df_1.iloc[i]["resume"][:6500]
                
                prompt = get_prompt(resume, label, self.tokenizer)
                if prompt:
                    data.append(prompt)
                if len(data) <= BATCH_SIZE:
                    break
            else:
                i = random.randint(0, len(self.df) - 1)
                label = self.df.iloc[i]["label"]
                resume = self.df.iloc[i]["resume"][:6500]
                prompt = get_prompt(resume, label, self.tokenizer)
                if prompt:
                    data.append(prompt)

        random.shuffle(data)  # Shuffle the data before returning

        return data



def prepare_batch(batch, tokenizer):
    inputs = []
    targets = []
    for i in range(len(batch)):
        inputs.append(batch[i][:-1])
        targets.append(batch[i][-1:])
    inputs_tokenized = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=False
    )
    outputs_tokenized = tokenizer(
        targets, return_tensors="pt", padding=True, truncation=False
    )

    return inputs_tokenized, outputs_tokenized.input_ids

def print_gpu_memory_usage():
    gpu_allocated = torch.cuda.memory_allocated(0)
    gpu_cached = torch.cuda.memory_reserved(0)
    print(f"GPU Memory Allocated: {gpu_allocated / (1024 ** 2)} MB")
    print(f"GPU Memory Cached: {gpu_cached / (1024 ** 2)} MB")

class CustomTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        test_dataset,
        num_epochs,
        test_every=100,
        test_batch_size=100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
       
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        weights = torch.ones((250880))
        weights[19] = 2
        weights[20] = 4
        self.criterion = nn.CrossEntropyLoss()

        self.test_every = test_every
        self.test_batch_size = test_batch_size
        self.highest_f1 = float('-inf')

    def find_accuracy(self, logits, targets):
        total = targets.shape[0]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        print('Target       = ', end="")
        for i in range(len(preds)):
            print(targets[i][0].item(),'    |   ', end="")
        print()

        print('Predicted    = ', end="")
        for i in range(len(preds)):
            print(preds[i].item() , '   |   ', end="")
        print()
        correct = (preds.view(-1) == targets.view(-1)).sum().item()
        return correct / total
    
    def find_predicted_labels(self, logits):
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        predicted_labels = torch.where(predicted_labels == 20, predicted_labels, torch.tensor([19]))
        return predicted_labels.tolist()

    def run_test_batch(self):
        self.model.eval()
        total_loss = 0
        accuracy = []
        curr_batch_no = 0
        y_pred_accumulated = []  # Accumulate predicted labels
        y_true_accumulated = []  # Accumulate true labels
        with torch.no_grad():
            for batch in tqdm(range(self.test_batch_size), desc="Calculting test accuracy and loss"):
                curr_batch_no += 1
                input_tokens, targets = prepare_batch(
                    self.test_dataset.get_next_batch(), self.tokenizer
                )
                self.optimizer.zero_grad()
                for key in input_tokens:
                    input_tokens[key] = input_tokens[key].to(self.device)

                logits = self.model(**input_tokens).logits
                logts = logits[:, -1:, :].view(-1, logits.shape[-1])
                logts = logts.to('cpu')
                def convert_values(list_values):
                    converted_values = [0 if value == 19 else 1 if value == 20 else value for value in list_values]
                    return converted_values
                
                # Accumulate predicted and true labels
                score_logts = convert_values(self.find_predicted_labels(logts))
                score_targets = convert_values(targets.view(-1).tolist())

                # Accumulate predicted and true labels
                y_pred_accumulated.extend(score_logts)
                y_true_accumulated.extend(score_targets)

                # log current and overall accuracy
                accuracy.append(self.find_accuracy(logts, targets))

                loss = self.criterion(logts, targets.view(-1))

                total_loss += loss.item()

                gc.collect()
                torch.cuda.empty_cache()

                if curr_batch_no >= self.test_batch_size:
                    self.model.train()
                    break
            overall_accuracy = accuracy_score(y_true_accumulated, y_pred_accumulated)
            overall_precision = precision_score(y_true_accumulated, y_pred_accumulated, pos_label=1)
            overall_recall = recall_score(y_true_accumulated, y_pred_accumulated, pos_label=1)
            overall_f1 = f1_score(y_true_accumulated, y_pred_accumulated,  pos_label=1)
            if  overall_f1 > self.highest_f1:
                self.highest_f1 = overall_f1
                with open("best_test_result.txt", "a+") as file:
                    file.write(f"Overall Accuracy: {overall_accuracy}\n")
                    file.write(f"Overall Precision: {overall_precision}\n")
                    file.write(f"Overall Recall: {overall_recall}\n")
                    file.write(f"Overall F1: {overall_f1}\n")
                    file.write(f"Overall Loss: {total_loss/curr_batch_no}\n")
                    file.write("\n\n")
            logging.info(
                f"Test: Avg. acc: {overall_accuracy:.4f}, Avg. prec: {overall_precision}, Avg. rec: {overall_recall}, Avg. f1: {overall_f1}, Overall Loss: {total_loss/curr_batch_no}"
            )

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            accuracy = []
            curr_batch_no = 0
            self.optimizer.zero_grad()
            
            y_pred_accumulated = []  # Accumulate predicted labels
            y_true_accumulated = []  # Accumulate true labels
            
            for batch in range(len(self.train_dataset) // BATCH_SIZE):
                curr_batch_no += 1
                input_tokens, targets = prepare_batch(
                    self.train_dataset.get_next_batch(), self.tokenizer
                )
                for key in input_tokens:
                    input_tokens[key] = input_tokens[key].to(self.device)

                logits = self.model(**input_tokens).logits


                logts = logits[:, -1:, :].view(-1, logits.shape[-1])
                logts = logts.to('cpu')

                for key in input_tokens:
                    input_tokens[key] = input_tokens[key].to('cpu')

                def convert_values(list_values):
                    converted_values = [0 if value == 19 else 1 if value == 20 else value for value in list_values]
                    return converted_values
                
                # Accumulate predicted and true labels
                score_logts = convert_values(self.find_predicted_labels(logts))
                score_targets = convert_values(targets.view(-1).tolist())

                # Accumulate predicted and true labels
                y_pred_accumulated.extend(score_logts)
                y_true_accumulated.extend(score_targets)

                # Log current and overall accuracy
                accuracy.append(self.find_accuracy(logts, targets))

                loss = self.criterion(logts, targets.view(-1))
                '''
                # Print current accuracy and loss
                logging.info(
                    f"Batch: {curr_batch_no} Train: Current accuracy: {accuracy[-1]}, Overall accuracy: {sum(accuracy)/len(accuracy)}, Loss: {loss.item()}"
                )
                '''
                # Calculate overall accuracy, precision, recall, and F1 score
                overall_accuracy = accuracy_score(y_true_accumulated, y_pred_accumulated)
                overall_precision = precision_score(y_true_accumulated, y_pred_accumulated, pos_label=1)
                overall_recall = recall_score(y_true_accumulated, y_pred_accumulated, pos_label=1)
                overall_f1 = f1_score(y_true_accumulated, y_pred_accumulated,  pos_label=1)
                logging.info(
                    f"Ep: {epoch+1}, B: {curr_batch_no} Train: Cur. loss: {loss.item():.4f}, Cur. acc: {accuracy[-1]:.4f}, Avg. acc: {overall_accuracy:.4f}, Avg. prec: {overall_precision}, Avg. rec: {overall_recall}, Avg. f1: {overall_f1}"
                )

                loss.backward()
                if curr_batch_no % OPTIMIZE_STEP == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

                gc.collect()
                torch.cuda.empty_cache()

                if curr_batch_no % self.test_every == 0:
                    
                    # Calculate overall accuracy, precision, recall, and F1 score
                    overall_accuracy = accuracy_score(y_true_accumulated, y_pred_accumulated)
                    overall_precision = precision_score(y_true_accumulated, y_pred_accumulated)
                    overall_recall = recall_score(y_true_accumulated, y_pred_accumulated)
                    overall_f1 = f1_score(y_true_accumulated, y_pred_accumulated)

                    logging.info(
                        f"Epoch: {epoch+1}, Overall Accuracy: {overall_accuracy}, Overall Precision: {overall_precision}, Overall Recall: {overall_recall}, Overall F1: {overall_f1}, Overall Loss: {total_loss/curr_batch_no}"
                    )


                    y_pred_accumulated.clear()
                    y_true_accumulated.clear()
                    accuracy = []
                    self.model.eval()
                    self.run_test_batch()
                    self.model.train()
            
            self.optimizer.step()




@click.command()
@click.option(
    "--dataset_json", help="Dataset json containing filename, resume and label"
)
def main(dataset_json):
    model, tokenizer = load_model()
    train, test = get_dataset(dataset_json)
    train_dataset = CustomDataset(train, tokenizer)
    test_dataset = CustomDataset(test, tokenizer, balanced_sampling=False)
    trainer = CustomTrainer(
        model,
        tokenizer,
        train_dataset,
        test_dataset,
        5,
        test_every=20,
        test_batch_size=10,
    )
    trainer.train()


if __name__ == "__main__":
    main()
