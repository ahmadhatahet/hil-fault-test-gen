SensorList = [
    (
        "Acceleration Pedal",
        "ACP",
        "Measures the amount of pressure applied to the accelerator pedal, indicating the driver's desired acceleration.",
    ),
    (
        "Wheel Steering Angle",
        "WSA",
        "Measures the angle at which the steering wheel is turned, determining the direction in which the vehicle is being steered.",
    ),
    (
        "Wheel Speed",
        "WS",
        "Measures the rotational speed of the vehicle's wheels, providing information on the vehicle's speed and potential wheel slippage.",
    ),
    (
        "Yaw Rate",
        "YR",
        "Measures the rate of rotation around the vertical axis of the vehicle, indicating its turning behavior and stability.",
    ),
    (
        "Steering Torque",
        "ST",
        "Measures the amount of force applied to the steering wheel, providing feedback on the driver's steering input and the vehicle's response.",
    ),
]

SensorsTemplate = """Full Name: {fn}
Short Name: {sn}
Description: {desc}
---
"""
Sensors_=""

for s in SensorList:
    Sensors_ += SensorsTemplate.format(fn=s[0],sn=s[1],desc=s[2])
    
Sensors = Sensors_[:-5]

if __name__ == "__main__":
    print(Sensors)