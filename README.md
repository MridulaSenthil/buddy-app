# buddy-app
# Buddy: Privacy-First AI Emotional Support System 🧠🤖

[![Streamlit App link here](https://buddy-app-huqdwdatvudwzqpb7jgcly.streamlit.app)

Buddy is an end-to-end conversational AI application designed for mental health monitoring. It leverages a dual-model architecture that splits workloads between localized, high-precision deep learning frameworks and cloud-based Large Language Models (LLMs) to maximize user privacy and minimize inference latency.

## 🚀 Key Features

- **Local Emotion Classification:** Fine-tuned `bert-base-uncased` transformer architecture executing 28-class multi-label emotion extraction locally on-device.
- **Dynamic Prompt Overrides:** Local keyword checking intercepts severe distress inputs to instantly trigger hardcoded crisis mitigation UI structures and force the generative pipeline into strict ethical support parameters.
- **Unified Full-Stack Python Framework:** A reactive user interface built completely with Streamlit, eliminating the architectural overhead of separate frontend (React) and backend (Node.js) deployments.
- **ETL Analytics Pipeline:** Automated asynchronous document parsing via PyMongo, pulling raw unstructured JSON strings into Pandas DataFrames, handling UTC-to-Local timezone localization, and loading aggregated mental trends into dynamic Plotly graphs.

## 🛠️ Technical Stack

- **Machine Learning Engine:** PyTorch, Hugging Face Transformers (`transformers`)
- **Generative Text Inference:** Llama 3.1 8B (via Groq Cloud LPU Architecture)
- **Database Engine:** MongoDB (Cloud Cluster via `pymongo` native driver)
- **Data Engineering & Visualization:** Pandas, NumPy, Plotly Express
- **Frontend & Deployment:** Streamlit Framework

## 📊 Model Training & Evaluation (BERT Pipeline)

The core classification model was fine-tuned on the **211,225-row GoEmotions dataset** (Kaggle). To address the deep class imbalances inherent to real-world behavioral language data, the following steps were implemented during training:
1. **Focal BCE Loss & Label Smoothing:** Applied custom optimization criteria to penalize simple majority class predictions and handle noisy training labels.
2. **Inverse-Frequency Weights:** Implemented mathematical scaling to boost minority class detection capabilities (e.g., *grief*, *nervousness*, *pride*).
3. **Metric Optimization:** Utilized **F1 Macro** as the absolute validation checkpoint criterion, avoiding the superficial accuracy inflation caused by the dominant "neutral" class.
