# -*- coding: utf-8 -*-
# !pip install datasets pandas torch transformers[torch] python-dotenv peft accelerate

import os
from dotenv import load_dotenv

load_dotenv()  # Load the .env file
hf_token = os.getenv("HF_TOKEN")

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import login
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.optim import AdamW

# Login to Hugging Face
login(token=hf_token)

# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
raw_data = load_dataset("tniranjan/aitamilnadu_tamil_stories_no_instruct", split="train[:1000]")
data = raw_data.train_test_split(train_size=0.95, seed=42)

# Preprocessing function
def preprocess_batch(batch):
    return tokenizer(
        batch["text"], 
        truncation=True, 
        padding=True, 
        max_length=200
    )

tokenized_dataset = data.map(
    preprocess_batch,
    batched=True,
    remove_columns=data["train"].column_names
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# ========== LORA CONFIGURATION ==========
lora_config = LoraConfig(
    r=8,                    # Rank (explained below)
    lora_alpha=32,          # Scaling factor (explained below)
    lora_dropout=0.1,       # Dropout for regularization
    bias="none",            # No bias parameters
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj", "c_fc"]  # Specific modules to apply LoRA
)
# ========================================

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Added missing save_strategy
    save_steps=500,
    learning_rate=1e-4,  # Increased from 1e-5 for fine-tuning
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=4,    # Increased batch size if GPU allows
    per_device_eval_batch_size=4,
    logging_steps=50,
    logging_dir="./logs",
    gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
    warmup_steps=100,  # Learning rate warmup
    fp16=True,  # Use mixed precision if GPU supports
    load_best_model_at_end=True,  # Load best model at the end
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    data_collator=data_collator,
    optimizers=(optimizer, None)  # Optimizer and scheduler
)

# Train
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_distilgpt2_Tamil")
tokenizer.save_pretrained("./fine_tuned_distilgpt2_Tamil")

# Load and test
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_distilgpt2_Tamil")
text = "ஒரு நாள் "
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(
    inputs.input_ids, 
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)
print(tokenizer.decode(output[0], skip_special_tokens=True))