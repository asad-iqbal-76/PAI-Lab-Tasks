from flask import Flask, render_template, request
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from datetime import datetime

app = Flask(__name__)

def init_chatbot():
    with open('restaurant_intents.json', 'r') as f:
        intents = json.load(f)
    questions = []
    answers = []
    for intent in intents:
        for pattern in intent['patterns']:
            questions.append(pattern)
            answers.append(intent['responses'][0])
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(questions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, questions, answers

model, index, questions, answers = init_chatbot()

chat_history = []  

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_message = request.form['message']
        chat_history.append({
            "sender": "You",
            "message": user_message,
            "time": datetime.now().strftime("%H:%M")
        })
        query_embedding = model.encode([user_message])
        distances, indices = index.search(query_embedding, k=1)
        bot_reply = answers[indices[0][0]]
        chat_history.append({
            "sender": "RestaurantBot",
            "message": bot_reply,
            "time": datetime.now().strftime("%H:%M")
        })
        
        return render_template('index.html', chat=chat_history)
    
    return render_template('index.html', chat=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
