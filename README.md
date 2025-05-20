# LLM-Based Chatbot Prototype for Car Reviews — Car-ing is Sharing

**An NLP chatbot prototype** for analyzing customer feedback at a car rental company, demonstrating modern language models' capabilities across 4 key tasks:

🔍 **Sentiment Analysis** | 🌍 **Machine Translation** | ❓ **Contextual Q&A** | 📝 **Smart Summarization**
---

## 🏆 Key Achievements

| Metric                  | Result   | Business Impact                          |
|-------------------------|----------|------------------------------------------|
| Sentiment Accuracy      | 0.8      | Auto-detection of negative reviews       |
| Translation BLEU Score  | 0.72     | Prepared for foreign market expansion    |
| F1 Score                | 0.85     | Balanced precision & recall              |
| Processing Time/Review  | <1s      | 90% reduction in manual analysis costs   |

---

## Key Takeaways

- Hands-on application of **LLMs** for solving multiple NLP subtasks in a single pipeline.
- Working with model outputs: handling **logits**, decoding spans, and understanding token IDs.
- How to **evaluate NLP models** using proper metrics like BLEU, F1, Accuracy and ROUGE.
- Integration of multiple pre-trained models into a production-like prototype.

---

## File Structure
```bash
project/
├── data/
│ ├── car_reviews.csv
│ └── reference_translations.txt
├── chatbot.py
├── requirements.txt
├── .gitignore
└── README.md
---
```
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