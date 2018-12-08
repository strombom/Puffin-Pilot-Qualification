
from flask import Flask
from flask import request
app = Flask(__name__)

import json
import pickle

# Adafruit motorkit requires permission to access I2C:
#  sudo usermod -a -G i2c www-data
from adafruit_motorkit import MotorKit
kit = MotorKit()

right_motor = kit.motor1
left_motor = kit.motor3


def save_config():
    config= {'throttle_left': throttle_left,
             'throttle_right': throttle_right}
    pickle.dump(config, open('/var/www/tmp/config.dat', 'wb'))

def load_config():
    try:
        data = pickle.load(open('/var/www/tmp/config.dat', 'rb'))
        config= {'throttle_left': data['throttle_left'],
                 'throttle_right': data['throttle_right']}
    except:
        config= {'throttle_left': 0.6,
                 'throttle_right': 0.6}
    return config

config = load_config()
throttle_left = config['throttle_left']
throttle_right = config['throttle_right']


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/forward')
def forward():
    left_motor.throttle = throttle_left
    right_motor.throttle = throttle_right
    return 'forward'

@app.route('/forward_left')
def forward_left():
    left_motor.throttle = throttle_left * 0.75
    right_motor.throttle = throttle_right
    return 'forward_left'

@app.route('/forward_right')
def forward_right():
    left_motor.throttle = throttle_left
    right_motor.throttle = throttle_right * 0.75
    return 'forward_right'

@app.route('/backward')
def backward():
    left_motor.throttle = -throttle_left
    right_motor.throttle = -throttle_right
    return 'backward'

@app.route('/left')
def left():
    left_motor.throttle = -throttle_left
    right_motor.throttle = throttle_right
    return 'left'

@app.route('/right')
def right():
    left_motor.throttle = throttle_left
    right_motor.throttle = -throttle_right
    return 'right'

@app.route('/stop')
def stop():
    left_motor.throttle = None
    right_motor.throttle = None
    return 'stop'

@app.route('/config', methods = ['GET', 'POST'])
def config():
    global throttle_left
    global throttle_right

    if request.method == 'GET':
        config = load_config()
        response = app.response_class(
            response = json.dumps(config),
            mimetype = 'application/json'
        )
        return response

    elif request.method == 'POST':
        left = request.args.get('throttle_left')
        right = request.args.get('throttle_right')
        if left is not None:
            try:
                throttle_left = float(left)
            except:
                pass
        if right is not None:
            try:
                throttle_right = float(right)
            except:
                pass

        save_config()
        return "OK"

if __name__ == '__main__':
    app.run()
