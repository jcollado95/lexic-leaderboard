import gradio as gr
import pandas as pd

def upload_file(file, preprocess):
    if not file:
        return "No file selected."
    
    with open(file, "r") as f:
        terms = f.readlines()

    if preprocess:
        return [
            gr.Textbox("Preprocessed terms:\n" + "".join(terms[:10]), label="Term preview (first 10 elements)"), 
            gr.Button(interactive=False)
        ]
    else:
        return [
            gr.Textbox("Not preprocessed terms:\n" + "".join(terms[-10:]), label="Term preview (last 10 elements)"), 
            gr.Button(interactive=True)
        ]

def clear_file():
    return [
        gr.Textbox("", label="Term preview"), 
        gr.Button(interactive=False)
    ]

def run_eval(model, vocab):
    return 0.5

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("Welcome to the Lexical Competence leaderboard.")
        with gr.Tabs(elem_classes="tab-buttons"):
            with gr.TabItem("Leaderboard"):
                leaderboard = gr.DataFrame("leaderboard.csv")

            with gr.TabItem("Evaluate your data"):
                gr.Markdown("Please submit a text file with a single term per line.")
                
                preprocess = gr.Checkbox(
                    label="Apply preprocessing", 
                    info="Check this to apply preprocessing to your data (i.e. lower string, non-alpha removal and stopword removal)"
                )

                with gr.Column():
                    file = gr.File()
                    vocab = gr.Textbox(label="Term preview")

                gr.Markdown("Select a model and click run to compute the score.")

                with gr.Column():
                    models = gr.Dropdown(
                        choices=["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"],
                        label="Model"
                    )
                run_btn = gr.Button(interactive=False)
                score = gr.Textbox(label="Score")

        file.upload(fn=upload_file, inputs=[file, preprocess], outputs=[vocab, run_btn])
        file.clear(fn=clear_file, outputs=[vocab, run_btn])
        run_btn.click(fn=run_eval, inputs=[models, vocab], outputs=[score])
    return demo

if __name__ == "__main__":
    models = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
    demo = build_demo()
    demo.queue(max_size=5) # Limits number of events
    demo.launch()