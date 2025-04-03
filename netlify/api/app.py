from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder='../public/static')

@app.route('/api/welcome', methods=['GET'])
def welcome():
    return jsonify({"message": "Welcome to the Flask API Service!"})

@app.route('/static/<path:path>', methods=['GET'])
def send_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)
