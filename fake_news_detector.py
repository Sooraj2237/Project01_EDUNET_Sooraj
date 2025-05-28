from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

def predict_fake_news(text, model, tokenizer, device, max_length=512):
    """
    Predicts whether a given text is fake or real news.

    Args:
        text (str): The news article text to classify.
        model (DistilBertForSequenceClassification): The trained model.
        tokenizer (DistilBertTokenizerFast): The trained tokenizer.
        device (torch.device): The device (cuda or cpu) to use for inference.
        max_length (int): The maximum length for tokenization.

    Returns:
        str: A message indicating whether the news is likely fake or real.
    """
    model.eval() # Set the model to evaluation mode

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=-1)

    if predictions.item() == 1:
        return "This news is likely real."
    else:
        return "This news is likely fake."


model_path = "./fake_news_detector_model"

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# For the usage let's ask the user for a title and text then combine it into content and predict a result.

title = input("Enter the title of the news: ")
text = input("Enter the text of the news: ")

content = "[TITLE] " + title + " [TEXT] " + text

result = predict_fake_news(content, model, tokenizer, device)
print(result)