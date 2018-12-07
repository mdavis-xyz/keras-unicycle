from flask import Flask
from flask import request
import json
app = Flask(__name__)

@app.route("/params", methods=["GET"])
def hello():
    payload = {"x":1,"y":2}
    return(json.dumps(payload))

@app.route("/frame", methods=["GET"])
def hello():
    payload = {"x":1,"y":2}
    return(json.dumps(payload))

if __name__ == "__main__":
    app.run(port=5002)
