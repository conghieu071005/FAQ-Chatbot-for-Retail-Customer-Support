
# FAQ Chatbot for Retail Customer Support

## 1. Project Overview

This project develops a **FAQ-based chatbot** designed to support customer inquiries for a retail store (e.g., electronics or mobile phone shops).
The chatbot is capable of **understanding Vietnamese user questions** and automatically providing appropriate answers based on a predefined FAQ knowledge base.

The system focuses on efficiently matching user questions with the most relevant existing answers using **Natural Language Processing (NLP)** techniques.

Specifically, the project:

* Accepts Vietnamese text queries from users
* Normalizes and preprocesses text (lowercasing, accent removal, stopword removal, synonym mapping)
* Converts questions into numerical vector representations
* Matches user queries against stored FAQ questions
* Returns the most relevant answer when the similarity score exceeds a defined threshold
* Rejects unclear or irrelevant questions to avoid incorrect responses

---

## 2. Data Description

The input data for this project is a **JSON file containing FAQ pairs**, where each entry includes:

* **question**: A frequently asked customer question (in Vietnamese)
* **answer**: The corresponding answer provided by the store

### Data Characteristics

* Unstructured **text-based data**
* Focused on:

  * Product information
  * Sales policies, warranties, and delivery
  * Common customer inquiries
* The data is used to:

  * Train a TF-IDF vectorizer
  * Store vector embeddings in a vector database for fast retrieval

---

## 3. Project Objectives

The main objectives of this project are to:

* Automate responses to frequently asked customer questions
* Reduce the workload of human customer support staff
* Improve customer experience through instant responses
* Apply **NLP techniques** to a real-world chatbot application
* Provide a scalable foundation for future intelligent chatbot systems

---

## 4. Models and Techniques Used

Instead of deep learning models, this project emphasizes **lightweight and interpretable NLP techniques**, including:

### 4.1. TF-IDF (Term Frequency – Inverse Document Frequency)

* Converts text into numerical feature vectors
* Measures the importance of words in each question
* Implemented manually to better support Vietnamese text

### 4.2. Cosine Similarity

* Measures similarity between user query vectors and FAQ vectors
* Higher scores indicate higher semantic similarity

### 4.3. Token Overlap Scoring

* Measures keyword overlap between user queries and FAQ questions
* Improves matching accuracy for short or informal inputs

### 4.4. Hybrid Similarity Scoring

* Combines **Cosine Similarity** and **Token Overlap**
* Reduces incorrect matches and improves robustness

### 4.5. ChromaDB (Vector Database)

* Stores TF-IDF vectors for FAQ questions
* Enables fast and scalable similarity search

---

## 5. Interface and Deployment

* The chatbot can be used via:

  * **Command-line interface (CLI)**
  * **Web-based interface built with Streamlit**
* The web interface includes:

  * Real-time chat interaction
  * Conversation history
  * Simple and user-friendly design

---

## 6. Summary

This project demonstrates a practical application of **NLP and information retrieval techniques** to build a real-world FAQ chatbot.
Despite not using deep learning, the system achieves effective performance within a limited knowledge domain and can be extended for future enhancements.

---

If you want, I can also:

* Shorten this README for a compact GitHub repository
* Add sections such as **Installation**, **Usage**, or **Project Structure**
* Rewrite it in a more academic or portfolio-oriented style
