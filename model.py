import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 加载 tokenizer
LORA_MODEL_PATH = "/root/autodl-tmp/MedGuideQG/lora_medical"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,  # 兼容 bitsandbytes 量化
    device_map="auto",
)

# 加载 LoRA 训练好的权重
if os.path.exists(LORA_MODEL_PATH):
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    print("✅ 加载 LoRA 训练模型成功！")
else:
    print("⚠️ 未找到 LoRA 训练模型，使用原始模型！")

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def generate_response(user_input):
    """生成 AI 回答"""
    messages = [
        {"role": "system", "content": "你是一位康复医学专家"},
        {"role": "user", "content": user_input},
    ]

    # 格式化为模型所需的格式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 转换为模型输入
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,  # 限制生成长度
        )

    # 提取生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码模型输出
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def interactive_medical_qna():
    """交互式三阶段医疗问答"""
    
    print("\n🩺 **您好，这里是智慧医疗小助手，请您描述您的病情。**\n")
    
    # **第一轮: 用户输入病情描述**
    patient_input = input("📝 患者: ")
    if patient_input.lower() == "exit":
        print("👋 退出问答系统。")
        return
    
    # **第一轮: 诊断 & 鉴别诊断**
    messages_round1 = [
        {"role": "system", "content": "你是一位康复医学专家"},
        {"role": "user", "content": f"你将根据患者在{patient_input}中提供的信息，仔细思考，给出1个可能的诊断。\n"
                                   "随后根据可能性从高到低，依次给出3至5个可能的鉴别诊断。\n"
                                   "请只回答诊断和鉴别诊断，不要在回答中包含其他内容。"}
    ]
    diagnosis_response = generate_response(patient_input)
    print("\n🔹 **第一轮 (诊断 & 鉴别诊断):**\n", diagnosis_response)

    # **第二轮: 机器生成提问**
    print(f"根据模型的诊断：{diagnosis_response}\n")
    print("系统提问: 根据诊疗指南，你能否更详细地描述自己的问题呢？\n")
    
    # **第二轮: 用户输入二轮问题**
    second_input = input("✏️ 患者: ")
    if second_input.lower() == "exit":
        print("👋 退出问答系统。")
        return
    
    # **第二轮: 诊疗指南 & 治疗方案**
    messages_round2 = [
        {"role": "system", "content": "你是一位康复医学专家"},
        {"role": "user", "content": f"根据上述回答，你考虑患者的诊断是{diagnosis_response}。\n"
                                   "这是这个疾病的诊疗指南{指南}，请根据指南回答:\n"
                                   "1. 患者病情分析\n"
                                   "2. 最终诊断\n"
                                   "3. 进一步检查及治疗方案\n"
                                   "请引用诊疗指南的原文。"}
    ]
    treatment_response = generate_response(second_input)
    print("\n🔹 **第二轮 (病情分析 & 治疗方案):**\n", treatment_response)

    # **第三轮: 机器生成提问**
    print(f" 根据模型的治疗建议：{treatment_response}\n")
    print("系统提问: 根据这个指南，您能回答这个是否符合自己的问题，是否有其他症状需要补充吗？\n")
    
    # **第三轮: 用户输入三轮问题**
    third_input = input("✏️ 患者: ")
    if third_input.lower() == "exit":
        print("👋 退出问答系统。")
        return
    
    # **第三轮: 医生角色总结 & 详细回答**
    messages_round3 = [
        {"role": "system", "content": "你是一位康复医学专家"},
        {"role": "user", "content": "请总结并以医生角色回答患者。"}
    ]
    final_response = generate_response(third_input)
    print("\n🔹 **第三轮 (医生详细回答患者):**\n", final_response)

    print("\n✅ **问答结束，感谢您的咨询！**")

if __name__ == "__main__":
    interactive_medical_qna()
