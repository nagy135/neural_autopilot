import pygame
import sys
import math
import numpy as np
import time
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.estimator import regression
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import lstm, simple_rnn 
from tflearn.data_utils import samplewise_zero_center


black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
blue = (0,0,255)

BRICK_SIZE = 100

initial_x = 50
initial_y = 984
initial_heading = 270

mymap2 = [
        [1,0,1,1,0,1,1,1,1,1],
        [1,0,0,1,0,1,0,0,0,0],
        [1,1,0,0,0,1,0,1,1,0],
        [0,0,0,1,1,1,0,0,1,0],
        [0,1,1,0,0,0,1,0,1,0],
        [0,0,0,0,1,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,0,0,0,1,0,0,0,1,0],
        [1,0,1,0,1,0,1,0,1,0],
        [1,0,1,0,0,0,1,0,0,0]
        ]
mymap = [
        [1,1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,1,1,0,0],
        [0,0,1,1,1,0,0,1,1,0],
        [0,1,0,0,0,1,0,0,0,0],
        [0,1,0,1,0,1,1,1,1,1],
        [0,0,0,1,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,0,0,1],
        [0,0,0,1,0,0,0,1,0,0],
        [0,1,0,1,0,1,0,1,1,0],
        [0,1,0,0,0,1,0,0,0,0]
        ]
def map_to_rectangles( mymap ):
    rectangles = {}
    i = 1
    for y in range(len(mymap)):
        for x in range(len(mymap[0])):
            if mymap[y][x] == 1:
                rectangles[i] = [(x*100,y*100),100,100]
                i += 1
    return rectangles

class Neuralpilot(object):
    def __init__(self, mymap):
        print('function: __init__')
        self.score = 0
        self.rectangles = map_to_rectangles(mymap)
        self.mymap = mymap


    def setup(self):
        print('function: setup')
        pygame.init()
        pygame.display.set_caption('Neural Network driver')

        self.display_width = 1000
        self.display_height = 1000
        self.crashed = False
        

        self.heading = initial_heading
        self.speed = 5
        self.x = initial_x
        self.y = initial_y
        self.memory = []

        self.gameDisplay = pygame.display.set_mode((self.display_width,self.display_height))
        self.clock = pygame.time.Clock()



    def draw_car(self,x,y):
        pygame.draw.circle(self.gameDisplay, blue, (int(x),int(y)), 5, 0)

    def draw_maze(self):
        for rectangle in self.rectangles.values():
            pygame.draw.rect(self.gameDisplay, black, (rectangle[0][0],rectangle[0][1],rectangle[1],rectangle[2]))

    def draw_laser(self, x, y):
        pygame.draw.line(self.gameDisplay, blue, (self.x, self.y), (x, y), 1)
        pygame.draw.line(self.gameDisplay, blue, (x-10, y-10), (x+10, y+10), 3)
        pygame.draw.line(self.gameDisplay, blue, (x-10, y+10), (x+10, y-10), 3)
    
    def is_maze_hit(self, x , y , goal_included=False):
        if goal_included or True:
            if y < 0:
                return True
        ##left/right
        if x <= 0 or x >=self.display_width or y >= self.display_height:
            return True
        ##left/right

        for rectangle in self.rectangles.values():
            temp_x = rectangle[0][0]
            temp_y = rectangle[0][1]
            temp_width = rectangle[1]
            temp_height = rectangle[2]
            if x >= temp_x and x <= temp_x + temp_width and y >= temp_y and y <= temp_y + temp_height:
                return True
        return False
        

    def is_won(self):
        if self.y <= 10:
            return True
        return False


    def pixel_to_coords (self, coords ):
        return (coords[0] // BRICK_SIZE), (coords[1] // BRICK_SIZE)
    def run(self, autopilot=False, model=None):
        print('function: run')
        self.setup()
        right = False
        left = False
        pause = False
        if not autopilot:
            try:
                save = np.load('data.npy')
            except FileNotFoundError:
                save = np.array([[[0, 0, 0, 0, 0, 0],[0,0,0]]])

        while not self.crashed:
            movement = [0,0]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.crashed = True
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    coords = self.pixel_to_coords(pos)
                    self.mymap[coords[1]][coords[0]] = (1 + self.mymap[coords[1]][coords[0]]) % 2
                    self.rectangles = map_to_rectangles(self.mymap)
                    self.gameDisplay.fill(white)
                    self.draw_maze()
                    pygame.display.update()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if not autopilot:
                            save = np.concatenate((save, self.memory))
                            np.save('data.npy', save)
                        self.crashed = True
                    if event.key == pygame.K_LEFT:
                        left = True
                    if event.key == pygame.K_UP:
                        pause = not pause
                    elif event.key == pygame.K_RIGHT:
                        right = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        left = False
                    elif event.key == pygame.K_RIGHT:
                        right = False
            if pause:
                continue

            self.gameDisplay.fill(white)
            self.draw_maze()


            ## get input neuron data

            distances = []
            ## right laser
            temp_heading = ( self.heading + 45 ) % 360
            temp_x = self.x
            temp_y = self.y
            temp_winner_x, temp_winner_y = self.shoot_laser(temp_x, temp_y, temp_heading, True)
            distances.append( self.euclidean_distance(self.x, self.y, temp_winner_x, temp_winner_y) )

            ## left laser
            temp_heading = ( self.heading - 45 ) % 360
            temp_x = self.x
            temp_y = self.y
            temp_winner_x, temp_winner_y = self.shoot_laser(temp_x, temp_y, temp_heading, True)
            distances.append( self.euclidean_distance(self.x, self.y, temp_winner_x, temp_winner_y) )

            ## front laser
            temp_heading = self.heading % 360
            temp_x = self.x
            temp_y = self.y
            temp_winner_x, temp_winner_y = self.shoot_laser(temp_x, temp_y, temp_heading, True)
            distances.append( self.euclidean_distance(self.x, self.y, temp_winner_x, temp_winner_y) )

            ## left laser
            temp_heading = ( self.heading - 90 ) % 360
            temp_x = self.x
            temp_y = self.y
            temp_winner_x, temp_winner_y = self.shoot_laser(temp_x, temp_y, temp_heading)
            distances.append( self.euclidean_distance(self.x, self.y, temp_winner_x, temp_winner_y) )

            ## right laser
            temp_heading = ( self.heading + 90 ) % 360
            temp_x = self.x
            temp_y = self.y
            temp_winner_x, temp_winner_y = self.shoot_laser(temp_x, temp_y, temp_heading)
            distances.append( self.euclidean_distance(self.x, self.y, temp_winner_x, temp_winner_y) )

            ## back laser
            temp_heading = ( self.heading + 180 ) % 360
            temp_x = self.x
            temp_y = self.y
            temp_winner_x, temp_winner_y = self.shoot_laser(temp_x, temp_y, temp_heading)
            distances.append( self.euclidean_distance(self.x, self.y, temp_winner_x, temp_winner_y) )
            if not autopilot:
                if left:
                    self.heading = (self.heading - 3) % 360
                    movement = [1,0,0]
                elif right:
                    self.heading = (self.heading + 3) % 360
                    movement = [0,1,0]
                else:
                    self.heading = self.heading
                    movement = [0,0,1]

                new_x, new_y = self.move(self.x, self.y, self.heading, self.speed)
 
                self.memory.append([distances, movement])
                ## get input neuron data
            else:
                predicted = model.predict(np.array(distances).reshape((-1, 6)))
                if np.argmax(predicted) == 0:
                    self.heading = (self.heading - 3) % 360
                elif np.argmax(predicted) == 1:
                    self.heading = (self.heading + 3) % 360
                else:
                    self.heading = self.heading

                new_x, new_y = self.move(self.x, self.y, self.heading, self.speed)

            if self.is_maze_hit(self.x, self.y):
                new_x = initial_x
                new_y = initial_y
                self.heading = initial_heading
            if self.is_won():
                new_x = initial_x
                new_y = initial_y
                self.heading = initial_heading
            self.x = new_x
            self.y = new_y
            self.draw_car(new_x, new_y)

            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()
        quit()

    def shoot_laser(self, x, y, heading, ignore_goal=False):
        passed_goal = False
        while True:
            if self.is_maze_hit(x, y):
                if ignore_goal and y <= 0:
                    passed_goal = True
                    break
                self.draw_laser(x, y)
                break

            x, y = self.move(x, y, heading, 5)
        if passed_goal:
            x, y = self.move(x, y, heading, 150)
            self.draw_laser(x, y)
        return x, y

    def is_in_wall(self, x, y):
        return True
    
    def move(self, x, y, heading, speed):
        return (x + math.cos(math.radians(heading)) * speed , y + math.sin(math.radians(heading)) * speed)

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )

    def run_nn(self):
        print('function: run_nn')
        pass

    def create_network(self):
        print('function: create_network')
        convnet = input_data(shape=[None, 6], name='input')
        convnet = embedding(convnet, input_dim=10000, output_dim=128)
        convnet = lstm(convnet, 128, dropout=0.8)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 1024, activation='relu')
        # convnet = conv_2d(convnet, 32, 5, activation='relu')
        # convnet = max_pool_2d(convnet, 5)

        # convnet = conv_2d(convnet, 32, 5, activation='relu')
        # convnet = max_pool_2d(convnet, 5)

        # convnet = conv_2d(convnet, 32, 5, activation='relu')
        # convnet = max_pool_2d(convnet, 5)

        # convnet = fully_connected(convnet, 1024, activation='relu')

        # convnet = dropout(convnet, 0.5)
        
        # convnet = fully_connected(convnet, 1024, activation='relu')

        # convnet = dropout(convnet, 0.6)

        # convnet = fully_connected(convnet, 1024, activation='relu')


        # convnet = embedding(convnet, input_dim=256, output_dim=256)
        # convnet = lstm(convnet, 256, dropout=0.8)
        # convnet = embedding(convnet, input_dim=256, output_dim=255)
        # convnet = lstm(convnet, 256, dropout=0.8)
        # convnet = embedding(convnet, input_dim=256, output_dim=255)
        # convnet = lstm(convnet, 256, dropout=0.8)
        # convnet = embedding(convnet, input_dim=256, output_dim=255)
        # convnet = lstm(convnet, 255, dropout=0.8)

        # convnet = simple_rnn(convnet, 6, dropout=0.8)
        # convnet = simple_rnn(convnet, 6, dropout=0.8)
        # convnet = simple_rnn(convnet, 6, dropout=0.8)
        # convnet = simple_rnn(convnet, 6, dropout=0.8)

        # convnet = fully_connected(convnet, 1024, activation='relu')
        # convnet = dropout(convnet, 0.8)
        # convnet = fully_connected(convnet, 1024, activation='relu')

        # convnet = fully_connected(convnet, 1024, activation='relu')
        # convnet = dropout(convnet, 0.8)
        # convnet = fully_connected(convnet, 1024, activation='relu')

        convnet = fully_connected(convnet, 3, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')
        return model

    def train_network(self, data):
        print('function: train_network')

        train = data[:-500]
        test = data[-500:]

        X = np.array([i[0] for i in train]).reshape(-1,6)
        Y = [i[1] for i in train]


        test_x = np.array([i[0] for i in test]).reshape(-1,6)
        test_y = [i[1] for i in test]


        model = self.create_network()

        model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=200, show_metric=True, run_id='neural_pilot')

        model.save("model.tfl")

    def even_out_data(self, data):
        hist = dict()
        for row in data:
            if np.argmax(row[1]) in hist:
                hist[np.argmax(row[1])] += 1
            else:
                hist[np.argmax(row[1])] = 1
        lowest_n = 99999999
        for key in hist:
            if hist[key] < lowest_n:
                lowest_n = hist[key]
        result = []
        for key in hist:
            hist[key] = lowest_n
        for row in data:
            if hist[np.argmax(row[1])] > 0:
                result.append(row)
                hist[np.argmax(row[1])] -= 1
        np_arr = np.array(result)
        np.random.shuffle(np_arr)
        return np_arr.tolist()
instance = Neuralpilot(mymap)
loaded_model = instance.create_network()
loaded_model.load('model.tfl')
instance.run(True, loaded_model)
# instance.run()

# loaded_data = np.load('data.npy')
# loaded_data = instance.even_out_data(loaded_data)
# instance.train_network(loaded_data)
