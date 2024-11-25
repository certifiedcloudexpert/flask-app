from flask import Flask
import openai

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is running!"

if __name__ == '__main__':
    app.run(debug=True)
