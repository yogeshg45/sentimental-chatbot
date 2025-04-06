import gradio as gr
from huggingface_hub import InferenceClient

# Load sentiment classification model
client = InferenceClient("distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment
def analyze_sentiment(message, system_message, max_tokens, temperature, top_p):
    result = client.text_classification(message)
    label = result[0]['label']
    score = round(result[0]['score'] * 100, 2)
    response = f"Sentiment: {label} ({score}%)"
    return response

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Sentiment Analysis Chatbot")

    with gr.Row():
        with gr.Column(scale=3):
            message = gr.Textbox(label="Enter your message")
            system_message = gr.Textbox(value="You are a sentiment analysis bot.", label="System message")
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")

            submit_btn = gr.Button("Analyze")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Sentiment Result", interactive=False)

    # Wire up function
    submit_btn.click(
        fn=analyze_sentiment,
        inputs=[message, system_message, max_tokens, temperature, top_p],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
