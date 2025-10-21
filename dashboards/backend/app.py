from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data')
def get_data():
    # In a real application, you would fetch and process your data here
    # For now, we'll just return some sample data
    sample_data = {
        'message': 'Hello from the Flask backend!',
        'items': [1, 2, 3, 4, 5]
    }
    return jsonify(sample_data)

if __name__ == '__main__':
    app.run(debug=True)
