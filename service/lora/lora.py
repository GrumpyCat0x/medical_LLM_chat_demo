import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_data(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def preprocess_data(data: list) -> tuple[list[str], list[str]]:
    # Assuming data is a list of dictionaries with 'input' and 'output' keys
    inputs = [item["question"] for item in data]
    outputs = [item["answer"] for item in data]
    return inputs, outputs


def tokenize_data(
    tokenizer: AutoTokenizer,
    inputs: list[str],
    outputs: list[str],
    max_length: int = 512,
) -> dict:
    model_inputs = tokenizer(
        inputs, max_length=max_length, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            outputs, max_length=max_length, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_dataset(
    tokenizer: AutoTokenizer, inputs: list[str], outputs: list[str]
) -> Dataset:
    tokenized_data = tokenize_data(tokenizer, inputs, outputs)
    dataset = Dataset.from_dict(tokenized_data)
    return dataset


def do_fine_tune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_path: str = "./data/medical_dialogue.json",
) -> None:
    # Load and preprocess data
    data = load_data(data_path)
    inputs, outputs = preprocess_data(data)
    dataset = create_dataset(tokenizer, inputs, outputs)

    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./temp",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()
