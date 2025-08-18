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

# Initialize database with Jase Medical data
cursor.execute("SELECT COUNT(*) FROM knowledge")
if cursor.fetchone()[0] == 0:
    initial_data = [
        "The Jase Case costs $289.95 and includes 10 medications: 5 life-saving antibiotics and 5 symptom relief meds, treating over 50 infections. It offers 28 add-on medication options and a KidCase for ages 2-11.",
        "Jase Daily provides an extended supply of prescription medications for conditions like diabetes, heart health, cholesterol, blood pressure, mental health, and family planning.",
        "Jase Go is a $129.95 travel-sized emergency med kit covering over 30 common travel conditions like traveler’s diarrhea, STIs, motion sickness, UTIs, and pneumonia.",
        "UseCase UTI costs $99.95 and is used to relieve urinary pain, treat lower urinary tract infections, vaginal candidiasis, and jock itch. It includes UTI test strips.",
        "UseCase Parasites starts at $199.95, with compounded Ivermectin and Mebendazole to treat parasitic infections inside and outside the body.",
        "Orders are reviewed by expert physicians, and medications are shipped directly from a licensed pharmacy. Contact support at answers@jase.com or (888) 522-6912.",
        "Jase Medical does not accept insurance but accepts HSA cards in some cases. Medications should be used within expiration dates and only when medical help is unavailable."
    ]
    for content in initial_data:
        cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
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

@app.after_request
def add_headers(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))