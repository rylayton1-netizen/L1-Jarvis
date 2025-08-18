import os
import sqlite3
from flask import Flask, request, render_template, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

app = Flask(__name__)

# Set up SQLite database
conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge
                  (id INTEGER PRIMARY KEY, content TEXT)''')
conn.commit()

def search_db(query):
    cursor.execute("SELECT content FROM knowledge WHERE content LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    return ' '.join([row[0] for row in results])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_ai():
    query = request.form['query']
    context = search_db(query)
    prompt = f"Based on this information: {context}\n\nAnswer the question for a phone support agent handling inquiries about Jase Medical emergency medications: {query}\nSuggest a professional script to say on the phone."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)