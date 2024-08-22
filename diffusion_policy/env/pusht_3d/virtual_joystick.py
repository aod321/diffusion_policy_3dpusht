import pygame
import sys
import socket as std_socket  # Import standard library's socket with alias
import zmq
import time

# Initialize Pygame
pygame.init()

# Screen size
screen_size = (800, 600)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Virtual Joystick")

# Color definitions
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Virtual joystick settings
joystick_center = (400, 300)  # Center of the joystick
joystick_radius = 75          # Radius of the outer joystick circle
knob_radius = 20              # Radius of the joystick knob
knob_position = joystick_center  # Initial position of the joystick knob

# Control state
dragging = False

# Frequency control
frequency = 10  # Hz
period = 1.0 / frequency  # Time period corresponding to the frequency
last_sent_time = 0

# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.PUB)
fixed_port = 5555  # Fixed port number
socket.bind(f"tcp://*:{fixed_port}")  # Bind to the fixed port

def draw_joystick():
    # Draw the background circle of the joystick
    pygame.draw.circle(screen, RED, joystick_center, joystick_radius, 5)
    # Draw the joystick knob
    pygame.draw.circle(screen, BLACK, knob_position, knob_radius)

def update_knob_position(mouse_pos):
    global knob_position
    # Calculate the vector from the joystick center to the mouse position
    dx = mouse_pos[0] - joystick_center[0]
    dy = mouse_pos[1] - joystick_center[1]
    distance = (dx**2 + dy**2) ** 0.5
    
    # If the distance is less than the joystick radius, move the knob to the mouse position
    if distance <= joystick_radius:
        knob_position = mouse_pos
    else:
        # Otherwise, limit the knob within the joystick boundary
        knob_position = (
            joystick_center[0] + joystick_radius * dx / distance,
            joystick_center[1] + joystick_radius * dy / distance,
        )

def get_normalized_position():
    # Calculate the normalized joystick position (-1 to 1)
    normalized_x = -((knob_position[0] - joystick_center[0]) / joystick_radius)
    normalized_y = -((knob_position[1] - joystick_center[1]) / joystick_radius)

    # Avoid negative zero
    normalized_x = 0. if normalized_x == -0.0 else normalized_x
    normalized_y = 0. if normalized_y == -0.0 else normalized_y

    return normalized_x, normalized_y

def send_control_data(position):
    # Convert the normalized position to a string and send via ZeroMQ
    message = f"{position[0]:.2f},{position[1]:.2f}"
    socket.send_string(message)

while True:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            socket.close()
            context.term()
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.math.Vector2(event.pos).distance_to(joystick_center) < joystick_radius:
                dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            knob_position = joystick_center  # Reset the knob to the center

        if event.type == pygame.MOUSEMOTION:
            if dragging:
                update_knob_position(event.pos)

    # Send control data at the specified frequency
    current_time = time.time()
    if current_time - last_sent_time >= period:
        normalized_position = get_normalized_position()
        print(f"Normalized Position: {normalized_position}")  # Print the normalized knob position
        send_control_data(normalized_position)
        last_sent_time = current_time

    draw_joystick()
    pygame.display.flip()
    pygame.time.Clock().tick(60)
