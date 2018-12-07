
from flask import Flask
app = Flask(__name__)

# Adafruit motorkit requires permission to access I2C:
#  sudo usermod -a -G i2c www-data
from adafruit_motorkit import MotorKit
kit = MotorKit()

left_motor = kit.motor1
right_motor = kit.motor3

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/forward')
def forward():
    left_motor.throttle = 0.7
    right_motor.throttle = 0.7
    return 'forward'

@app.route('/backward')
def backward():
    left_motor.throttle = -0.7
    right_motor.throttle = -0.7
    return 'backward'

@app.route('/left')
def left():
    left_motor.throttle = 0.5
    right_motor.throttle = -0.5
    return 'left'

@app.route('/right')
def right():
    left_motor.throttle = -0.5
    right_motor.throttle = 0.5
    return 'right'

@app.route('/stop')
def stop():
    left_motor.throttle = None
    right_motor.throttle = None
    return 'stop'

if __name__ == '__main__':
    app.run()
