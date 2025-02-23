import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import bitsandbytes as bnb
import os

# 数据集路径
TRAIN_DATA_PATH = "/root/autodl-tmp/MedGuideQG/OurData/data/train.json"
DEV_DATA_PATH = "/root/autodl-tmp/MedGuideQG/OurData/data/dev.json"
TEST_DATA_PATH = "/root/autodl-tmp/MedGuideQG/OurData/data/test.json"

# 加载数据集
dataset = load_dataset("json", data_files={
    "train": TRAIN_DATA_PATH,
    "dev": DEV_DATA_PATH,
    "test": TEST_DATA_PATH
})

# 划分数据集：训练集、开发集和测试集
train_dataset = dataset["train"]
dev_dataset = dataset["dev"]
test_dataset = dataset["test"]

# LoRA 训练配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 仅微调注意力层
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 选择模型
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 兼容 bitsandbytes 量化
    device_map="auto",
)

# 适配 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenization
def tokenize_function(batch):
    src_texts = batch["src"]
    tgt_texts = batch["tgt"]

    # 使用 Tokenizer 进行批量处理
    model_inputs = tokenizer(
        [src + " " + tgt for src, tgt in zip(src_texts, tgt_texts)],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # 添加 labels
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

# **应用 Tokenization**
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["src", "tgt"])
dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["src", "tgt"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["src", "tgt"])

# **训练参数**
training_args = TrainingArguments(
    output_dir='/root/autodl-tmp/MedGuideQG/lora_medical',
    per_device_train_batch_size=2, # 确保训练批次
    per_device_eval_batch_size=2,  # 确保评估批次
    gradient_accumulation_steps=4, # 确保梯度累积
    num_train_epochs=3, # 确保训练轮数
    logging_dir='/root/autodl-tmp/MedGuideQG/logs',
    logging_strategy='steps',
    logging_steps=10, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_bnb_8bit",
    report_to="tensorboard",  # 启用 TensorBoard 日志记录
)

# **创建 Trainer**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
)

# **开始训练**
trainer.train()

# **保存模型**
model.save_pretrained('/root/autodl-tmp/MedGuideQG/lora_medical')
tokenizer.save_pretrained('/root/autodl-tmp/MedGuideQG/lora_medical')

print("✅ LoRA 训练完成，模型已保存！")
