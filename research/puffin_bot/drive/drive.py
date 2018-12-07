
from flask import Flask
app = Flask(__name__)

from adafruit_motorkit import MotorKit
kit = MotorKit()


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/forward')
def forward():
    return 'forward'

@app.route('/backward')
def backward():
    return 'backward'

if __name__ == '__main__':
    app.run()
