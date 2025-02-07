# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 模型名称
# model_name = "Qwen/Qwen2.5-7B-Instruct"

# # 加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# # 加载模型，支持 8-bit 量化减少显存占用
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# # 设备选择（GPU 优先）
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(DEVICE)

# def chat_with_model(user_input):
#     """ 根据用户输入生成医疗问题的回答 """
#     messages = [
#         {"role": "system", "content": "You are Qwen, a medical assistant. Provide helpful and accurate medical information."},
#         {"role": "user", "content": user_input},
#     ]
    
#     # 格式化为模型所需的格式
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     # 转换为模型输入
#     model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

#     # 生成回答
#     with torch.no_grad():
#         generated_ids = model.generate(
#             **model_inputs,
#             max_new_tokens=256,  # 限制生成长度
#         )
    
#     # 仅提取生成的内容
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     # 解码模型输出
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response

# # 交互式问答循环
# if __name__ == "__main__":
#     print("💡 医疗问答系统已启动，输入问题以获取答案（输入 'exit' 退出）")
#     while True:
#         user_input = input("🩺 你: ")  # 用户输入
#         if user_input.lower() in ["exit", "quit", "退出"]:
#             print("👋 退出问答系统。")
#             break
        
#         response = chat_with_model(user_input)
#         print(f"🤖 AI: {response}\n")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def generate_response(messages, max_tokens=512):
    """ 生成模型回答 """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def interactive_medical_qna():
    """ 交互式三阶段医疗问答 """
    
    print("\n💡 欢迎使用智能医疗问答系统，请按提示输入信息。输入 'exit' 退出。\n")

    # **第一轮: 用户输入病情描述**
    patient_input = input("🩺 请描述您的病情: ")
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
    diagnosis_response = generate_response(messages_round1)
    print("\n🔹 **第一轮 (诊断 & 鉴别诊断):**\n", diagnosis_response)

    # **第二轮: 用户输入二轮问题**
    print("\n💡 现在，请基于上面的诊断，输入您想进一步咨询的问题。")
    second_input = input("✏️ 请输入您的问题: ")
    if second_input.lower() == "exit":
        print("👋 退出问答系统。")
        return
    
    # **第二轮: 诊疗指南 & 治疗方案**
    messages_round2 = [
        {"role": "system", "content": "你是一位康复医学专家"},
        {"role": "user", "content": f"根据上述回答，你考虑根据这位患者{patient_input}描述的内容，考虑患者的诊断是{diagnosis_response}。\n"
                                    "这是这个疾病的诊疗指南{指南}，你必须根据指南，去回答以下内容:\n"
                                    "1. 患者病情的分析\n"
                                    "2. 最终考虑的诊断\n"
                                    "3. 进一步检查及治疗方案\n"
                                    "在回答过程中，请引用诊疗指南的原文。"}
    ]
    treatment_response = generate_response(messages_round2)
    print("\n🔹 **第二轮 (病情分析 & 治疗方案):**\n", treatment_response)
    
    # **第三轮: 用户输入三轮问题**
    print("\n💡 现在，请输入您的最终咨询问题，让医生给您一个完整的总结。")
    third_input = input("✏️ 请输入您的问题: ")
    if third_input.lower() == "exit":
        print("👋 退出问答系统。")
        return

    # **第三轮: 医生角色总结 & 详细回答**
    messages_round3 = [
        {"role": "system", "content": "你是一位康复医学专家"},
        {"role": "user", "content": f"根据上述回答的内容，进行总结，并以医生的角色详细的回答患者。\n"
                                    "请使用专业的医学用语，确保患者可以理解，并提供详细的治疗建议。"}
    ]
    final_response = generate_response(messages_round3)
    print("\n🔹 **第三轮 (医生详细回答患者):**\n", final_response)

if __name__ == "__main__":
    interactive_medical_qna()
