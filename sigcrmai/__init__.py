from flask import Flask, request

app = Flask(__name__)

from sigcrmai import routes

if __name__ == "_main_":
    app.run(host='0.0.0.0', port=5000)