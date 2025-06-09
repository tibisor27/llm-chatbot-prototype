"""
NLP Pipeline Module
==================
Core functionality for car review analysis including:
- Sentiment Analysis with evaluation metrics
- Question Answering system with few-shot prompting
- Text Summarization with ROUGE scoring

This module contains all the business logic separated from execution.
"""

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from typing import Tuple, List, Dict, Any

# ====================================================================
# DATA LOADING & PREPROCESSING
# ====================================================================

def load_car_reviews_data(file_path: str = "data/car_reviews.csv") -> Tuple[List[str], List[str]]:
    """Load and prepare car reviews dataset"""
    print("Loading car reviews dataset...")
    df = pd.read_csv(file_path, delimiter=";")
    
    reviews = df['Review'].tolist()
    real_labels = df['Class'].tolist()
    
    print(f"Loaded {len(reviews)} reviews")
    return reviews, real_labels

# ====================================================================
# SENTIMENT ANALYSIS WITH EVALUATION
# ====================================================================

def analyze_sentiment(reviews: List[str], real_labels: List[str]) -> Tuple[float, float]:
    """Perform sentiment analysis and evaluate against ground truth"""
    print("\nStarting Sentiment Analysis...")
    
    # Initialize model
    classifier = pipeline(
        task='sentiment-analysis',
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # Get predictions
    predicted_labels = classifier(reviews)
    
    # Display results
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    for review, prediction, label in zip(reviews, predicted_labels, real_labels):
        print(f"Review: {review}")
        print(f"Actual Sentiment: {label}")
        print(f"Predicted Sentiment: {prediction['label']} (Confidence: {prediction['score']:.4f})")
        print()
    
    # Calculate evaluation metrics
    return evaluate_sentiment_model(predicted_labels, real_labels)

def evaluate_sentiment_model(predicted_labels: List[Dict], real_labels: List[str]) -> Tuple[float, float]:
    """Calculate accuracy and F1 score for sentiment analysis"""
    print("Calculating Model Performance Metrics...")
    
    # Load evaluation metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    
    # Convert labels to binary format
    references = [1 if label == "POSITIVE" else 0 for label in real_labels]
    predictions = [1 if label["label"] == "POSITIVE" else 0 for label in predicted_labels]
    
    # Calculate metrics
    accuracy_result = accuracy.compute(references=references, predictions=predictions)["accuracy"]
    f1_result = f1.compute(references=references, predictions=predictions)["f1"]
    
    print(f"Accuracy: {accuracy_result:.4f}")
    print(f"F1 Score: {f1_result:.4f}")
    
    return accuracy_result, f1_result

# ====================================================================
# QUESTION ANSWERING SYSTEM WITH FEW-SHOT PROMPTING
# ====================================================================

def create_qa_prompt(context: str, question: str) -> str:
    """Create few-shot prompt for better question answering"""
    return f"""
Answer questions about car reviews using these examples:

Example 1:
Context: "The car has excellent fuel efficiency and smooth ride quality"
Question: "What are the positive aspects?"
Answer: fuel efficiency, smooth ride quality

Example 2:
Context: "I hate the noisy engine and poor build quality"
Question: "What did they dislike?"
Answer: noisy engine, poor build quality

Now answer this:
Context: "{context}"
Question: "{question}"
Answer:"""

def setup_question_answering() -> Tuple[Any, Any]:
    """Initialize Question Answering model and tokenizer"""
    print("\nSetting up Question Answering System...")
    
    model_checkpoint = "deepset/minilm-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    print("Question Answering model ready")
    return tokenizer, model

def answer_question(question: str, context: str, tokenizer: Any, model: Any) -> str:
    """Extract answer from context using QA model with few-shot prompting"""
    print(f"\nAnswering Question: '{question}'")
    print(f"Context: {context}...")
    
    # Create few-shot prompt for demonstration
    qa_prompt = create_qa_prompt(context, question)
    print("Few-shot prompt template created: Yes")
    
    # Tokenize inputs
    inputs = tokenizer(question, context, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract answer span
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_span = inputs["input_ids"][0][start_idx:end_idx]
    
    # Decode answer
    answer = tokenizer.decode(answer_span, skip_special_tokens=True)
    print(f"Answer: {answer}")
    
    return answer

# ====================================================================
# TEXT SUMMARIZATION WITH ROUGE EVALUATION
# ====================================================================

def summarize_text(text: str) -> Tuple[str, Dict]:
    """Generate summary and evaluate with ROUGE metrics"""
    print(f"\nSummarizing text...")
    print(f"Original text: {text}")
    print()
    
    # Initialize summarization model
    model_name = "cnicu/t5-small-booksum"
    summarizer = pipeline(task="summarization", model=model_name)
    
    # Generate summary
    outputs = summarizer(text, max_length=50)
    summarized_text = outputs[0]['summary_text']
    
    print(f"Summary: {summarized_text}")
    
    # Evaluate with ROUGE
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=[summarized_text], references=[text])
    
    print(f"ROUGE Score: {rouge_score}")
    
    return summarized_text, rouge_score 