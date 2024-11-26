from transformers import pipeline  # This is to load the Hugging Face model
from flask import Flask, request, jsonify  # Added request and jsonify for handling requests and JSON

# Initialize the question-answering pipeline with a pre-trained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is running!"

# Define a route to handle the question and provide an answer
@app.route('/ask', methods=['POST'])
def ask_question():
    # Get the question from the incoming JSON request
    data = request.get_json()
    question = data.get("question")
    context = data.get("context")
    
    # If either the question or context is missing, return an error message
    if not question or not context:
        return jsonify({"error": "Both 'question' and 'context' are required"}), 400
    
    # Use the pre-trained model to get an answer
    result = qa_pipeline({"question": question, "context": context})
    
    # Return the answer as a JSON response
    return jsonify({"answer": result["answer"]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
