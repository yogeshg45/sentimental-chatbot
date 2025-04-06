import gradio as gr
from huggingface_hub import InferenceClient

# Load sentiment classification model
client = InferenceClient("distilbert-base-uncased-finetuned-sst-2-english")

def respond(message, history, system_message, max_tokens, temperature, top_p):
    # Perform sentiment analysis on the message
    result = client.text_classification(message)
    label = result[0]['label']
    score = round(result[0]['score'] * 100, 2)
    response = f"Sentiment: {label} ({score}%)"
    history.append((message, response))
    return history

# Gradio Chat Interface with additional (but unused) controls
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a sentiment analysis bot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
    title="Sentiment Analysis Chatbot",
    description="Enter a message and get the sentiment prediction.",
)

if __name__ == "__main__":
    demo.launch()
