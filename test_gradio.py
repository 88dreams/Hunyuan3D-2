#!/usr/bin/env python3

import gradio as gr

def simple_test(text):
    return f"Gradio is working! Input: {text}"

# Create a simple interface
demo = gr.Interface(
    fn=simple_test,
    inputs=gr.Textbox(label="Test Input"),
    outputs=gr.Textbox(label="Output"),
    title="Test Interface"
)

if __name__ == "__main__":
    print("Starting simple Gradio test...")
    demo.launch(server_port=7860, share=False)
