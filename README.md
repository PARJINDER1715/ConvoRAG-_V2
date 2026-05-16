# ConvoRAG v3

## Project Overview

ConvoRAG v3 is an AI chat application built using Python and Flask.
It uses:

* Flask for backend API
* SQLite database for storing chat sessions
* Groq API for AI responses
* CSV data for conversations
* Machine Learning model for intent classification

The project can:

* Save chat history
* Detect user intent
* Manage conversation sessions
* Use cached AI responses to reduce API cost
* Handle contradiction detection

---

# Folder Structure

```bash
convorag_v3/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python libraries
├── .env                    # API key file
├── conversations.csv       # Conversation dataset
├── static/
│   └── index.html          # Frontend page
├── models/
│   └── intent_model.pkl    # Trained ML model
├── core/
│   ├── db.py               # Database functions
│   ├── persona_engine.py   # Persona handling
│   ├── intent_classifier.py# Intent prediction
│   ├── conflict_resolver.py# Contradiction detection
│   └── __init__.py
```

---

# Features

✅ AI Chat System

✅ Session Management

✅ SQLite Database Support

✅ Intent Classification

✅ Contradiction Detection

✅ Groq API Integration

✅ Cached Responses

---

# Requirements

Install Python libraries:

```bash
pip install -r requirements.txt
```

---

# Setup Instructions

## Step 1: Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Add Groq API Key

Open `.env` file and add:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Step 4: Run the Project

```bash
python app.py
```

---

# Open in Browser

```bash
http://localhost:5000
```

---

# Main API Endpoints

| Method | Endpoint                 | Description             |
| ------ | ------------------------ | ----------------------- |
| GET    | /api/sessions            | Get all sessions        |
| GET    | /api/sessions/<id>       | Get session messages    |
| DELETE | /api/sessions/<id>       | Delete session          |
| GET    | /api/groq/cache/stats    | Cache statistics        |
| GET    | /api/contradiction_pairs | Get contradiction pairs |
| POST   | /api/contradiction_pairs | Add contradiction pair  |

---

# Technologies Used

* Python
* Flask
* SQLite
* Pandas
* Machine Learning
* HTML/CSS
* Groq API

---

# Future Improvements

* Better UI design
* User authentication
* Real-time chat
* Deployment on cloud
* Advanced AI memory system

---

# Author

Parjinder Singh
