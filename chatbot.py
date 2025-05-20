import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.metrics import accuracy_score, f1_score
import evaluate
file_path = "data/car_reviews.csv"
df = pd.read_csv(file_path, delimiter=";")

reviews = df['Review'].tolist()
real_labels = df['Class'].tolist()

classifier = pipeline(task='sentiment-analysis',model="distilbert-base-uncased-finetuned-sst-2-english") # sentiment analysis model
predicted_labels = classifier(reviews)

for review, prediction, label in zip(reviews, predicted_labels, real_labels):
    print(f"Review: {review}\nActual Sentiment: {label}\nPredicted Sentiment: {prediction['label']} (Confidence: {prediction['score']:.4f})\n")
    
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
references = [1 if label == "POSITIVE" else 0 for label in real_labels]
prediction = [1 if label["label"] == "POSITIVE" else 0 for label in predicted_labels]

accuracy_result_dict = accuracy.compute(references=references, predictions=prediction)
accuracy_result = accuracy_result_dict["accuracy"]
print(f"Accuracy: {accuracy_result}")

f1_result_dict = f1.compute(references=references, predictions=prediction)
f1_result = f1_result_dict["f1"]
print(f"F1 Score: {f1_result}")

first_review = reviews[0]

translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-es") # translation model
translated_review = translator(first_review, max_length=27)[0]['translation_text'] # translate the first review 
print(f"Original Review: {first_review}")
print(f"Translated Review: {translated_review}")

with open("data/reference_translations.txt", 'r') as file:
    lines = file.readlines()
references = [line.strip() for line in lines]
print(f"Spanish translation references:\n{references}")

bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=[translated_review], references=[references])
print(f"BLEU Score: {bleu_score['bleu']}")

model_ckp = "deepset/minilm-uncased-squad2" 
tokenizer = AutoTokenizer.from_pretrained(model_ckp) #used for tokenizing the question and context
model = AutoModelForQuestionAnswering.from_pretrained(model_ckp) # question answering model
context = reviews[1]
print(f"Context: {context}")
question = "What did he like about the brand?"
inputs = tokenizer(question, context, return_tensors="pt") # tokenize the question and context
with torch.no_grad():
    outputs = model(**inputs) # pass the inputs(unpacked) to the model

start_idx = torch.argmax(outputs.start_logits) # find the index of the start of the answer
end_idx = torch.argmax(outputs.end_logits) + 1 # find the index of the end of the answer
answer_span = inputs["input_ids"][0][start_idx:end_idx] # get the answer
# Decode and show answer
answer = tokenizer.decode(answer_span)
print("Answer: ", answer)

#Summarize the last review in the dataset
text_to_summarize = reviews[-1]
print(f"Text to summarize: {text_to_summarize}")

model = "cnicu/t5-small-booksum"
summarizer = pipeline(task="summarization", model=model)
outputs = summarizer(text_to_summarize, max_length=50)
summarized_text = outputs[0]['summary_text']
print(f"Summarized text:\n{summarized_text}")
rouge = evaluate.load("rouge")
rouge_score = rouge.compute(predictions=[summarized_text], references=[text_to_summarize])
print(f"ROUGE Score:\n{rouge_score}")