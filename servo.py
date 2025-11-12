import lgpio
import time


servo_pin = 18 

frequency = 50
period = 1_000_000 // frequency  # microseconds per cycle

def set_servo_angle(angle):
    chip = lgpio.gpiochip_open(0) 
    # angle (0–180°) to pulse width (1000–2000 µs)
    pulse_width = int(1000 + (angle / 180) * 1000)

    #duty cycle as % of period
    duty_cycle = (pulse_width / period) * 100

    lgpio.tx_pwm(chip, servo_pin, frequency, duty_cycle)
    print(f"Servo angle: {angle}° | Duty cycle: {duty_cycle:.2f}%")
    time.sleep(0.5)

try:
    lgpio.gpio_claim_output(chip, servo_pin)

    set_servo_angle(180)
    time.sleep(2)

    set_servo_angle(0)
    time.sleep(1)

finally:
    # Stop PWM and clean up
    lgpio.tx_pwm(chip, servo_pin, 0, 0)
    lgpio.gpiochip_close(chip)
