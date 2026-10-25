import gradio as gr

def greet(name):
    return f"Hello {name}!"

gr.Interface(
    lambda x: x,
    "text",
    "text"
).launch(share=True)

print("Starting with share=True")
# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo = gr.Interface(
    fn=lambda x: f"Hello {x}",
    inputs="text",
    outputs="text"
)
# demo.launch()
# demo.launch(share=True)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    debug=True
)