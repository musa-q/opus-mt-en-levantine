from flask import Flask, render_template
from routes import bp

app = Flask(__name__)

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True)