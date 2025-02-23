import gradio as gr
from model import answer_question

# Gradio 界面定义
def answer_question(user_input, chat_history):
    return answer_question(user_input, chat_history)

# 创建 Gradio 接口
interface = gr.Interface(
    fn=answer_question,
    inputs=[gr.Textbox(lines=2, placeholder="请输入您的病情或问题...", label="用户提问"), gr.State([])],  # 输入问题并保留聊天历史
    outputs=[gr.Chatbot(), gr.State([])],  # 输出聊天历史并更新
    title="三阶段医疗智能问答系统",
    description="请描述您的病情，模型将给出答复。",
)

# 启动 Gradio 应用
interface.launch()
