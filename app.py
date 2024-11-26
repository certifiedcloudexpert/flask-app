from transformers import pipeline  # For the Hugging Face model
from flask import Flask, request, jsonify  # For Flask handling requests and JSON responses
from flask_lti import LTI  # For LTI support

# Load the pre-trained model for question answering (DistilBERT)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

app = Flask(__name__)

# LTI Route - Here we are adding the new LTI launch endpoint
@app.route('/lti_launch', methods=['POST'])
def lti_launch():
    data = request.get_json()  # Get the LTI launch data
    # You'd need to validate the LTI request here (this is just a placeholder)
    if not validate_lti_request(data):  # Validate the request (you need to implement this)
        return jsonify({"error": "Invalid LTI request"}), 400

    # If LTI request is valid, we send back a success message
    return jsonify({"message": "LTI request validated successfully"}), 200

# Home route to check if Flask is working
@app.route('/')
def home():
    return "Hello, Flask is running!"

# Define the route to handle the question and provide an answer
@app.route('/ask', methods=['POST'])
def ask_question():
    # Get the question and context from the incoming JSON request
    data = request.get_json()
    question = data.get("question")
    context = data.get("context")

    # If either the question or context is missing, return an error
    if not question or not context:
        return jsonify({"error": "Both 'question' and 'context' are required"}), 400

    # Use the pre-trained model to answer the question
    result = qa_pipeline({"question": question, "context": context})

    # Return the answer as a JSON response
    return jsonify({"answer": result["answer"]})

# Function to validate the LTI request (this is a placeholder for your validation)
def validate_lti_request(data):
    # Implement the validation logic here
    # For example, check that data contains a valid consumer key, signature, etc.
    # For now, this is just a placeholder returning True
    return True

if __name__ == '__main__':
    app.run(debug=True, port=5001)
