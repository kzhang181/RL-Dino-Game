# Kenneth Zhang
# CSC580
# Final Project

import gymnasium as gym
import turtle
import random
import time
import math
import numpy as np
from PIL import Image

HEIGHT = 100        # number of steps vertically from wall to wall of screen
WIDTH = 100         # number of steps horizontally from wall to wall of screen
PIXEL_H = 50        # pixel height + border on both sides
PIXEL_W = 50        # pixel width + border on both sides

SLEEP = 0.2     # time to wait between steps

GAME_TITLE = 'Chrome Dino'
BG_COLOR = 'white'

# Load images
img = Image.open("/Users/kennethzhang/Downloads/CSC 580 Final Project/images/Dino1.png")
img.save("dino.gif")
img = Image.open("/Users/kennethzhang/Downloads/CSC 580 Final Project/images/cactus.png")
img.save("cactus.gif")
img = Image.open("/Users/kennethzhang/Downloads/CSC 580 Final Project/images/DinoDucking1.png")
img.save("dinoDuck.gif")
img = Image.open("/Users/kennethzhang/Downloads/CSC 580 Final Project/images/Ptero1.png")  # New image for pterodactyl
img.save("ptero.gif")

class DinoGameEnv(gym.Env):
    def __init__(self, human=False, display_game=True, env_info={'state_space': None}):
        self.action_space = 3  # 0 = do nothing, 1 = jump, 2 = duck
        self.state_space = 12  
        self.human = human
        self.display = display_game

        # Setup Screen
        self.screen = turtle.Screen()
        self.screen.title("Dino Game")
        self.screen.bgcolor("white")
        self.screen.setup(width=800, height=300)
        self.screen.tracer(0)

        # Adding image shapes
        self.screen.addshape("dino.gif")
        self.screen.addshape("dinoDuck.gif")
        self.screen.addshape("cactus.gif")
        self.screen.addshape("ptero.gif")

        # Dino Image
        self.screen.register_shape("dino.gif")  # Ensure file exists
        self.dino = turtle.Turtle()
        self.dino.shape("dino.gif")
        self.dino.penup()
        self.dino.goto(-300, -50)
        self.dino.dy = -50

        # Score display
        self.score = 0
        self.score_display = turtle.Turtle()
        self.score_display.penup()
        self.score_display.hideturtle()
        self.score_display.goto(320, 100)  # Position at the top-right
        self.update_score()

        # Create obstacles list
        self.obstacles = []

        # Game Variables
        self.gravity = -1
        self.jump_power = 10
        self.is_jumping = False
        self.is_ducking = False
        self.stepper = 0

    def update_score(self):
        ' Updates the score display '
        self.score_display.clear()
        self.score_display.write(f'Score: {self.score}', align="center", font=("Arial", 20, "bold"))
    
    def jump(self):
        ' Dinosaur jumping '
        if not self.is_jumping:
            self.dino.shape("dino.gif")
            self.dino.dy = self.jump_power # Jump power
            self.is_jumping = True

    def duck(self):
        ' Ducks dinosaur '
        if not self.is_ducking and not self.is_jumping:
            self.dino.shape("dinoDuck.gif")
            self.dino.goto(-300, -60)
            self.is_ducking = True

    def unduck(self):
        ''' Unduck the dinosaur '''
        if self.is_ducking:
            self.dino.shape("dino.gif")
            self.dino.goto(-300, -50)
            self.is_ducking = False

    def step(self, action):
        ' Performs step action and returns the state, reward, done, and score '
        done = False
        if action == 1 and self.dino.ycor() == -50 and not self.is_jumping and not self.is_ducking: # Jump actions
            self.jump()
        elif action == 2 and not self.is_ducking and not self.is_jumping: # Duck action
            self.duck()
            self.stepper = 0
        elif self.is_ducking:
            self.stepper += 1
            if self.stepper >= 8: # Unducking after 8 steps
                self.unduck()

        if self.dino.ycor() > -50: # If dino is above ground, gravity will pull dino down
            self.dino.dy += self.gravity

        new_y = self.dino.ycor() + self.dino.dy # Update y-coord

        if new_y < -50 and not self.is_ducking: # If dino is below ground, it will reset the level
            new_y = -50
            self.dino.dy = 0  # Stop velocity when it hits the ground
            self.is_jumping = False  # Reset jumping flag when on the ground
        self.dino.sety(new_y)

        reward, done = self.move_obstacles() # Moves the obstacles and returns the rewards
    
        # Updates the score top right
        self.score += 1
        self.update_score()
        self.render()

        if done: 
            self.reset()

        state = self.get_state()
        return state, reward, done, self.score

    def get_state(self):
        ' State: dino y-coord, dino velocity, regular state?, dino ducking?,'
        '        dino jumping?, first obstacle passed dinosaur?, obstacle close?, second obstacle close to first?,'
        '        obstacle type(0 for cactus, 1 for pterodactly), obstacle duckable, obstacle too high?, obstacle position on dino x position? '

        if len(self.obstacles) > 0:
            firstObstacleX = self.obstacles[0].xcor() # Gets obstacle x coordinates
            firstObstacleY = self.obstacles[0].ycor() # Gets obstacle y coordinates

            if  -180 <= firstObstacleX <= -160: # Checks if the obstacle is close
                obstacle_close = 1
            else: obstacle_close = 0

            if firstObstacleX < -330: # Checks if the obstacle passed the dino
                dinoPassed = 1
            else: dinoPassed = 0

            if self.obstacles[0].shape() == "cactus.gif": # Checks for next obstacle type
                obstacle_type = 0
            else: obstacle_type = 1

            if firstObstacleY > -50: # Checks if the obstacle is above ground level
                obstacle_height = 1
            else: obstacle_height = 0

            if firstObstacleY  >= 20: # Checks if it is too high to jump or duck
                too_high = 1
            else: too_high = 0

            if -300 <= firstObstacleX <= -270: # Checks if the obstacle is on the exact same position as the dino
                obstacle_on_dino = 1
            else: obstacle_on_dino = 0
        else:
            obstacle_close = 0
            dinoPassed = 0
            obstacle_height = 0
            dinoPassed = 0
            too_high = 0
            obstacle_type = 0
            obstacle_on_dino = 0
        
        if self.dino.ycor() == -50: # Checks if the dino is on the ground
            dinoY = 0
        else: dinoY = 1

        if self.dino.dy == 0: # Checks if the dino is still falling
            dinoVelocity = 0
        else: dinoVelocity = 1

        if len(self.obstacles) > 1:
            if self.obstacles[1].xcor() - firstObstacleX < 75:
                second_close = 1
            else: second_close = 0
        else: second_close = 0

        return [dinoY, dinoVelocity, int((not(self.is_ducking) and not(self.is_jumping))), int(self.is_ducking),
                    int(self.is_jumping), dinoPassed, obstacle_close, second_close,
                    obstacle_type, obstacle_height, too_high, obstacle_on_dino]

    def spawn_obstacle(self):
        ' Randomly spawn either a cactus or a pterodactyl '
        if random.choice([True, False]):  # True: cactus, False: pterodactyl
            # Spawn a cactus
            obstacle = turtle.Turtle()
            obstacle.shape("cactus.gif")
            obstacle.penup()
            obstacle.goto(400, -50)
            self.obstacles.append(obstacle) 
        else:
            # Spawn a pterodactyl
            obstacle = turtle.Turtle()
            obstacle.shape("ptero.gif")
            obstacle.penup()
            # Position pterodactyl at a random height
            y_position = random.choice([-50, -25, 20])
            obstacle.goto(400, y_position)
            self.obstacles.append(obstacle)

    
    def move_obstacles(self):
        ' Move obstacles and check for collisions '
        done = False
        reward = 1
        if len(self.obstacles) == 0:
            self.spawn_obstacle()
        
        # Move all obstacles
        for obstacle in self.obstacles:
            obstacle.setx(obstacle.xcor() - 12)  # Move obstacle to the left by 12 every step

            # Respawn obstacles when they move off-screen
            if obstacle.xcor() <= -400:
                self.obstacles.remove(obstacle)
                obstacle.hideturtle()

            # Gives reward of -100 if the dino hits the obstacle
            if self.check_collision(self.dino, obstacle):
                done = True
                reward = -100

            # If dino passed, gives reward of 20
            elif -320 <= obstacle.xcor() <= -300:
                reward = 20
            # Gives reward of 5 if it gets close to obstacle
            elif abs(obstacle.xcor() - self.dino.xcor()) < 80:
                reward = 5

            # Prevents excessive jumping    
            elif self.is_jumping:
                reward = -2
            elif self.is_ducking:
                reward = -2
        return reward, done

    def check_collision(self, dino, obstacle):
        ' Checks for dino collision with obstacle '
        distance_x = abs(dino.xcor() - obstacle.xcor())
        distance_y = abs(dino.ycor() - obstacle.ycor())

        if distance_x < 30 and distance_y < 30:  # Obstacles have a hit box of 30
            return True
        return False

    def respawn_obstacle(self, obstacle):
        ' Respawn obstacle '
        self.spawn_obstacle()

    def render(self):
        ' Updates screen display '
        try:
            self.screen.update()
        except turtle.Terminator:
            # Do nothing if the Turtle window is closed
            pass
        time.sleep(SLEEP)


    def reset(self):
        ' Resets game after collision '
        self.score = 0
        self.update_score()
        self.dino.goto(-300, -50)
        self.dino.dy = -50
        self.is_jumping = False
        self.is_ducking = False

        # Hides all current obstacles and respawn new one
        start_x = 400
        for obstacle in self.obstacles:
            obstacle.hideturtle() 
        self.obstacles.clear()
        self.spawn_obstacle()

        return self.get_state()

    def close(self):
        ' Closes game'
        print("Game closing")
        self.screen.bye()