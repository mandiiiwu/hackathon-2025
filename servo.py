import lgpio
import time

SERVO_PIN = 18  # Choose your desired GPIO pin (e.g., GPIO18)
PWM_FREQ = 50   # Standard servo PWM frequency (50 Hz)

    # Open the GPIO chip
h = lgpio.gpiochip_open(0)

    # Start PWM on the chosen pin
    # The duty cycle for servos is typically defined by pulse width in microseconds
    # A 50Hz signal has a period of 20,000 microseconds (1,000,000 / 50)
    # Common pulse widths for 0, 90, and 180 degrees are around 500, 1500, and 2500 us
lgpio.gpio_start_pwm(h, SERVO_PIN, PWM_FREQ, 0) # Start with 0% duty cycle initially

def set_servo_angle(angle):
        # Convert angle to pulse width in microseconds
        # This is a simplified linear mapping; adjust values for your specific servo
    min_pulse_width = 500  # Microseconds for 0 degrees
    max_pulse_width = 2500 # Microseconds for 180 degrees
    pulse_width = min_pulse_width + (angle / 180.0) * (max_pulse_width - min_pulse_width)

        # Calculate duty cycle percentage
    duty_cycle_percentage = (pulse_width / (1_000_000 / PWM_FREQ)) * 100

    lgpio.gpio_set_dutycycle(h, SERVO_PIN, duty_cycle_percentage)

try:
    while True:
            # Move servo to 0 degrees
        set_servo_angle(0)
        time.sleep(1)

            # Move servo to 90 degrees
        set_servo_angle(90)
        time.sleep(1)

            # Move servo to 180 degrees
        set_servo_angle(180)
        time.sleep(1)

except KeyboardInterrupt:
    # Stop PWM and close the GPIO chip on exit
    lgpio.gpio_stop_pwm(h, SERVO_PIN)
    lgpio.gpiochip_close(h)
