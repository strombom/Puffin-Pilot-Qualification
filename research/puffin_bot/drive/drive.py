
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

try:
    data = pickle.load(open('drive.dat', 'rb'))
    throttle_left = data['throttle_left']
    throttle_right = data['throttle_right']
except:
    throttle_left = 0.6
    throttle_right = 0.6


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/forward')
def forward():
    left_motor.throttle = throttle_left
    right_motor.throttle = throttle_right
    return 'forward'

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

@app.route('/stop')
def stop():
    left_motor.throttle = None
    right_motor.throttle = None
    return 'stop'

def save_config():
    config= {'throttle_left': throttle_left,
             'throttle_right': throttle_right}
    pickle.dump(config, open('drive.dat', 'wb'))

@app.route('/throttle', methods = ['GET', 'POST'])
def throttle():
    if request.method == 'GET':
        return json.dumps({'left': throttle_left,
                           'right': throttle_right})

    elif request.method == 'POST':
        left = request.args.get('left')
        right = request.args.get('right')
        if left is not None:
            throttle_left = int(left)
        if right is not None:
            throttle_left = int(right)

        save_config()
        return "OK"

if __name__ == '__main__':
    app.run()
