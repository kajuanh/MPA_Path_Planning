import pygame
import sys
import os

sys.path.append(os.path.abspath("/"))
from modify_code import MPA, Node, distance
import time
import random as rd
start_time = time.time()
mpa_obj = MPA('Test/map15_3.txt')
print(map_size := mpa_obj.map_size)
print(goals := mpa_obj.goals)
environment = mpa_obj.environment

dis_sol, final_sol, a_prey = mpa_obj.way([0, 0], [0, 12])


BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
YELLOW = (215, 225, 88)
GRAY = (150, 150, 100)
BLOCKSIZE = 40
WINDOW_WIDTH = WINDOW_HEIGHT = map_size*BLOCKSIZE


def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)
    moving_rect = pygame.Rect(0, 0, BLOCKSIZE-1, BLOCKSIZE-1)
    final_sol_index = 1
    limit = len(final_sol)-1
    speed = 0.1
    limit_line = 0
    st = final_sol[final_sol_index-1]
    en = final_sol[final_sol_index]
    x_speed = ((en[0] - st[0])*BLOCKSIZE)*0.1
    y_speed = ((en[1] - st[1])*BLOCKSIZE)*0.1
    while True:
        SCREEN.fill(BLACK)
        drawGrid()
        drawLineFinal(final_sol)
        pygame.draw.rect(SCREEN, (0, 255, 0), moving_rect)
        pygame.display.flip()
        if final_sol_index <= limit:
          if limit_line < 0.9:
            moving(moving_rect, x_speed, y_speed)
            limit_line += speed
          else:
            if final_sol_index < limit:
              final_sol_index += 1
              st = final_sol[final_sol_index-1]
              en = final_sol[final_sol_index]
              x_speed = ((en[0] - st[0])*BLOCKSIZE)*0.1
              y_speed = ((en[1] - st[1])*BLOCKSIZE)*0.1
              limit_line = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        CLOCK.tick(30)


def drawLineFinal(final_sol: list):
    for i in range(len(final_sol)-1):
        x1 = (final_sol[i][0]+0.5)*BLOCKSIZE
        y1 = (final_sol[i][1]+0.5)*BLOCKSIZE
        x2 = (final_sol[i+1][0]+0.5)*BLOCKSIZE
        y2 = (final_sol[i+1][1]+0.5)*BLOCKSIZE
        pygame.draw.line(SCREEN, (0, 0, 255), (x1, y1), (x2, y2), 2)


def drawGrid():
    for x in range(map_size):
        for y in range(map_size):
            rect = pygame.Rect(x*BLOCKSIZE, y*BLOCKSIZE,
                               BLOCKSIZE-1, BLOCKSIZE-1)
            if environment[x][y] == 1:
              pygame.draw.rect(SCREEN, GRAY, rect)
            elif [x, y] in [[0, 0], [0, 12]]:
              pygame.draw.rect(SCREEN, YELLOW, rect)
            else:
              pygame.draw.rect(SCREEN, WHITE, rect, 1)


def moving(moving_rect, x_speed, y_speed):
    moving_rect.x += x_speed
    moving_rect.y += y_speed
    pygame.draw.rect(SCREEN,(0,255,0), moving_rect)
    
main()
