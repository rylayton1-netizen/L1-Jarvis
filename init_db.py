import sqlite3
import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Jase Medical test data based on jasemedical.com
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
conn.close()
print("Database initialized with Jase Medical data!")