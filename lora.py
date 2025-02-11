import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from datasets import load_dataset
import os

# **数据集路径**
DATASET_PATH = "/root/autodl-tmp/MedGuideQG/OurData/MyData.json"

# **检查数据文件是否存在**
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ 数据文件 {DATASET_PATH} 未找到，请检查路径！")

# **加载数据集**
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# **划分数据集：80% 训练，20% 评估**
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# **LoRA 训练配置**
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 仅微调注意力层
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# **选择模型**
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# **修正 `bitsandbytes` 量化方式**
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # **兼容 `bitsandbytes`**
    device_map="auto",
)

# **适配 LoRA**
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# **修正 Tokenization（确保 `labels` 存在）**
def tokenize_function(batch):
    """ 处理 `src` 和 `tgt`，确保批处理时格式正确，并添加 `labels` """
    src_texts = batch["src"]
    tgt_texts = batch["tgt"]

    # **确保 `src` 和 `tgt` 都是字符串**
    src_texts = [" ".join(text) if isinstance(text, list) else text for text in src_texts]
    tgt_texts = [" ".join(text) if isinstance(text, list) else text for text in tgt_texts]

    # **使用 Tokenizer 进行批量处理**
    model_inputs = tokenizer(
        [src + " " + tgt for src, tgt in zip(src_texts, tgt_texts)],  # 拼接输入
        padding="max_length",  # **修正：确保所有样本长度一致**
        truncation=True,  # **修正：避免超长文本**
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,  # **修正：避免 `pyarrow` 格式错误**
        return_token_type_ids=False,  # **修正：确保输出一致**
    )

    # **添加 `labels`（用于计算 loss）**
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

# **应用 Tokenization**
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["src", "tgt"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["src", "tgt"])

# **训练参数**
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/MedGuideQG/lora_medical",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,  # **确保评估批次**
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="/root/autodl-tmp/MedGuideQG/logs",  # **TensorBoard 日志路径**
    logging_strategy="steps",  # **每 N 步记录一次**
    logging_steps=10,  # **每 10 步记录一次 loss**
    evaluation_strategy="epoch",  # **每个 `epoch` 进行评估**
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_bnb_8bit",
    report_to="tensorboard"  # **启用 `TensorBoard` 记录 loss**
)

# **创建 Trainer**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # **提供训练数据**
    eval_dataset=eval_dataset,  # **提供评估数据**
    tokenizer=tokenizer,
)

# **开始训练**
trainer.train()

# **保存模型**
model.save_pretrained("/root/autodl-tmp/MedGuideQG/lora_medical")
tokenizer.save_pretrained("/root/autodl-tmp/MedGuideQG/lora_medical")

print("✅ LoRA 训练完成，模型已保存！")
