# LLM-Based Chatbot Prototype for Car Reviews — Car-ing is Sharing

This project is a **prototype of a chatbot application** built for a car sales and rental company. The goal was to explore the capabilities of **Large Language Models (LLMs)** in real-world NLP tasks: **sentiment analysis**, **translation**, **question answering**, and **summarization**.

---

##  Technologies & Libraries

-  **Hugging Face Transformers** (`pipeline`, `AutoModelForQuestionAnswering`, `AutoTokenizer`)
-  **scikit-learn** for F1 & accuracy
-  **evaluate** library (Hugging Face metrics)
-  **pandas**
-  **PyTorch** for extractive QA inference

---

## 📌 Tasks Completed

###  1. Sentiment Analysis
- Used `distilbert-base-uncased-finetuned-sst-2-english` to classify 5 car reviews.
- Computed **accuracy** and **F1 score**:

###  2. English → Spanish Translation
- Used `Helsinki-NLP/opus-mt-en-es` model to translate.
- Evaluated translation quality using the **BLEU score** metric.

###  3. Extractive Question Answering
- Used `deepset/minilm-uncased-squad2` model
- Defined a custom question and provided relevant context extracted from customer reviews.

###  4. Summarization
- Summarized the final review using `cnicu/t5-small-booksum`.
- Evaluated summarization quality using the **ROUGE score** metric.
---

## File Structure

project/
├── data/
│ ├── car_reviews.csv
│ └── reference_translations.txt
├── chatbot.py
├── requirements.txt
├── .gitignore
└── README.md
---

## What I Learned

- Hands-on application of **LLMs** for solving multiple NLP subtasks in a single pipeline.
- Working with model outputs: handling **logits**, decoding spans, and understanding token IDs.
- How to **evaluate NLP models** using proper metrics like BLEU, F1, Accuracy and ROUGE.
- Integration of multiple pre-trained models into a production-like prototype.

---

##  Getting Started
1. Clone the repo:
```bash
git clone https://github.com/username/car-llm-chatbot.git
cd car-llm-chatbot

Install dependencies:

pip install -r requirements.txt

Run the chatbot script:
python chatbot.py

---

## 📄 License

This project is licensed under the MIT License.