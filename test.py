linear_velocity = 1
angular_velocity = -1

wheel_radius = 0.098
wheel_seperation = 0.0352   # 0.37558  0.0352 


left_wheel_velocity = (linear_velocity - (angular_velocity * wheel_seperation / 2)) / wheel_radius
right_wheel_velocity = (linear_velocity + (angular_velocity * wheel_seperation / 2)) / wheel_radius
print(f"Left Wheel Velocity: {left_wheel_velocity}, Right Wheel Velocity: {right_wheel_velocity}")