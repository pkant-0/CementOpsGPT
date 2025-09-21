import gradio as gr
from transformers import pipeline

# Load models
optimizer = pipeline("text-generation", model="gpt2")
assistant = pipeline("text2text-generation", model="google/flan-t5-base")

# Kiln Optimization Function
def suggest_kiln_settings(cement_type, raw_material):
    prompt = f"Suggest optimal kiln settings for {cement_type} cement using {raw_material}."
    result = optimizer(prompt, max_length=100, do_sample=True)[0]['generated_text']
    return result

# Troubleshooting Assistant Function
def troubleshoot(query):
    prompt = f"Troubleshooting guide for: {query}"
    result = assistant(prompt)[0]['generated_text']
    return result

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧱 CementOpsGPT – GenAI for Cement Manufacturing")

    with gr.Tab("🔥 Kiln Optimization"):
        cement_type = gr.Textbox(label="Cement Type (OPC/PPC)")
        raw_material = gr.Textbox(label="Raw Material Composition")
        output = gr.Textbox(label="Suggested Kiln Settings")
        btn = gr.Button("Generate Settings")
        btn.click(fn=suggest_kiln_settings, inputs=[cement_type, raw_material], outputs=output)

    with gr.Tab("💬 Troubleshooting Assistant"):
        query = gr.Textbox(label="Describe the Issue")
        response = gr.Textbox(label="AI Guidance")
        btn2 = gr.Button("Get Help")
        btn2.click(fn=troubleshoot, inputs=query, outputs=response)

demo.launch()
