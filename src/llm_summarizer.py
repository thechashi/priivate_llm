import click
import torch
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
import toml
import re
import traceback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


def get_model_tokenizer(model_name, adapters_name):
    model_name = model_name
    adapters_name = adapters_name

    print(f"Starting to load the model {model_name} into memory")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1

    stop_token_ids = [0]
    """
    return model, tokenizer


def get_summary(
    model, tokenizer, prompt, tokenizer_max_length=1500, min_len=50, max_len=400
):
    try:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=tokenizer_max_length,
            truncation=True,
        )
        # inputs.pop('token_type_ids')
        inputs.to("cuda")
        # generated_ids = model.generate(**inputs, min_length=min_len, max_new_tokens=max_len, num_beams=2, num_return_sequences=2)
        generated_ids = model.generate(
            **inputs, min_length=min_len, max_new_tokens=max_len
        )
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        substring = "Sure, here is a summary of the resume you provided:"
        output = outputs[len(prompt) :].strip()
        '''output = outputs[outputs.find(substring) + len(substring) :].strip()
        

        overall_index = output.find("Overall")
        for i in range(overall_index, len(output)):
            if output[i] == "\n":
                output = output[:i]
                break'''

        print("Input length: {}, Output length: {}".format(len(prompt), len(outputs)))
    except Exception as e:
        traceback.print_exc()
        output = ""
    return output


def get_resume_prompt(intro_blurb, resume):
    # prompt = """{} {} \n Summary:""".format(intro_blurb, resume)
    prompt = (
        """{} {}. \nSure, here is a summary of the resume you provided: \n """.format(
            intro_blurb, resume
        )
    )
    return prompt


def batch_resume_summarizer(
    model,
    tokenizer,
    intro_blurb,
    input_folder,
    output_folder,
    tokenizer_max_length=1500,
    min_len=50,
    max_len=400,
):
    # Define the path of the folder containing the text files
    input_folder = Path(input_folder)

    # Define the path of the folder where the output files will be saved
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over all files in the input folder
    for file_path in tqdm(input_folder.glob("*Resume*")):
        print("Processing: {}".format(file_path.name))
        # Read the text file
        with file_path.open("r") as file:
            text = file.read()
            if len(text) > 6500:
                text = text[:6500]
        prompt = get_resume_prompt(intro_blurb, text)
        # Process the text using the function
        summary = get_summary(
            model, tokenizer, prompt, tokenizer_max_length, min_len, max_len
        )

        # Save the processed text in a new file in the output folder
        output_filename = output_folder / file_path.name.split(".")[0]
        with output_filename.open("w", encoding="utf-8") as file:
            file.write(summary)


def file_count(folder):

    folder_path = Path(folder)

    file_count = 0
    for file_path in folder_path.glob("*"):
        if file_path.is_file():
            file_count += 1

    return file_count


@click.command()
@click.option("--input_folder", default=None, help="Folder containing resume texts")
@click.option("--output_folder", default=None, help="Folder to save the summaries")
@click.option(
    "--config_file_path",
    default="src/configs/summarizer.toml",
    help="Path to config file",
)
def main(input_folder, output_folder, config_file_path):
    config = toml.load(config_file_path)
    intro_blurb = config["meta"]["intro_blurb"]
    model_name = config["model"]["model_id"]
    adapters_name = config["meta"]["adapter_path"]

    print("Loading mdoel and tokenizer..")
    model, tokenizer = get_model_tokenizer(model_name, adapters_name)

    total_resumes = file_count(input_folder)
    print(
        "Producing resume summaries. Total number of resume files: {}".format(
            total_resumes
        )
    )
    batch_resume_summarizer(model, tokenizer, intro_blurb, input_folder, output_folder)


if __name__ == "__main__":
    main()
