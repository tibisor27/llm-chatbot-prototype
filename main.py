#!/usr/bin/env python3
import evaluate
import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from nlp_pipeline import (
    load_car_reviews_data,
    analyze_sentiment,
    setup_question_answering,
    answer_question,
    summarize_text
)

def main():
    
    try:
        # 1. Load data
        reviews, real_labels = load_car_reviews_data()
        
        # 2. Sentiment Analysis with evaluation
        accuracy, f1_score = analyze_sentiment(reviews, real_labels)
        
        # 3. Question Answering demonstration
        tokenizer, qa_model = setup_question_answering()
        context = reviews[1]  # Use second review as context
        question = "What did he like about the brand?"
        answer = answer_question(question, context, tokenizer, qa_model)
        
        # 4. Text Summarization with ROUGE evaluation
        text_to_summarize = reviews[-1]  # Use last review
        summary, rouge_scores = summarize_text(text_to_summarize)
        
        # 5. Final summary
        print(f" Final Results:")
        print(f"   • Sentiment Analysis Accuracy: {accuracy:.4f}")
        print(f"   • Sentiment Analysis F1 Score: {f1_score:.4f}")
        print(f"   • Question Answered: '{question}' → '{answer}'")

    except FileNotFoundError:
        print(" Error: Could not find data/car_reviews.csv")
        print("Please ensure the data file exists in the correct location.")
        sys.exit(1)
        
    except Exception as e:
        print(f" An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 