import pygame
from run import GameController
from constants import *
import pygame
import neat
import os
import time
import pickle
import keyboard
import math
import datetime
import threading
import multiprocessing
import concurrent.futures


class pacman_game:
    def __init__(self):
        self.game = GameController()
        self.game.startGame()
        self.pacman = self.game.pacman
        self.ghosts = self.game.ghosts  # list
        self.score = self.game.score
        self.clock = pygame.time.Clock()
        self.turning_points = self.game.nodes.nodesLUT.keys()
        self.intersection_points = [[], []]
        self.last_position = None  # Store the last position of the agent
        self.last_position_time = None
        for node in self.turning_points:
            self.intersection_points[0].append(node[0])
            self.intersection_points[1].append(node[1])

            # (node.position.x, node.position.y)
        # print(self.intersection_points[0])
        # print(self.intersection_points[1])

    # print("pacman at")
    # print(self.pacman.position)

    def test_ai(self):
        run = True
        # clock = pygame.time.Clock()
        self.game.startGame()
        while run:
            # clock.tick(600)

            self.game.loop()
            self.game.render()
            pygame.display.update()
            game_info = self.game.loop()
            # print(game_info.pacman_pos) #prints pacmans coordinates on the board
            # print(game_info.ghosts_pos) #prints all ghosts coordinates on the board

    def test_ai(self, net):
        """
        Test the AI against a human player by passing a NEAT neural network
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            game_info = self.game.loop()
            clock.tick(60)
            pacman_pos = (game_info.pacman_pos_x.x, game_info.pacman_pos_x.y)
            for node_pos in self.turning_points:
                if pacman_pos == node_pos:

                    output1 = net.activate((game_info.pacman_pos_x.x, game_info.pacman_pos_x.y,
                                            game_info.ghosts_pos[0].x - game_info.pacman_pos_x.x,
                                            game_info.ghosts_pos[0].y - game_info.pacman_pos_x.y,
                                            game_info.ghosts_pos[1].x - game_info.pacman_pos_x.x,
                                            game_info.ghosts_pos[1].y - game_info.pacman_pos_x.y,
                                            game_info.ghosts_pos[2].x - game_info.pacman_pos_x.x,
                                            game_info.ghosts_pos[2].y - game_info.pacman_pos_x.y,
                                            game_info.ghosts_pos[3].x - game_info.pacman_pos_x.x,
                                            game_info.ghosts_pos[3].y - game_info.pacman_pos_x.y,
                                            game_info.score
                                            ))
                    decision = output1.index(max(output1))
                    if decision == 0:
                        # self.game.pacman.direction = UP
                        keyboard.press('w')
                        keyboard.release('a')
                        keyboard.release('s')
                        keyboard.release('d')
                        # print("up")

                    elif decision == 1:
                        keyboard.press('s')
                        keyboard.release('d')
                        keyboard.release('a')
                        keyboard.release('w')
                        # print("down")

                    elif decision == 2:
                        keyboard.press('a')
                        keyboard.release('d')
                        keyboard.release('s')
                        keyboard.release('w')
                        # print("left")

                    elif decision == 3:
                        keyboard.press('d')
                        keyboard.release('s')
                        keyboard.release('a')
                        keyboard.release('w')

                        # print("right")
                else:
                    pass

    def train_ai(self, genome1, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        #net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        self.game.startGame()
        # self.pause.setPause(playerPaused=True)
        self.game.pause.setPause(playerPaused=False)
        time_stamp = None
        last_position = None  # Store the last position of the agent
        last_position_time = None

        while run:
            dt = self.clock.tick(300) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.pacman.update(dt)
            game_info = self.game.loop()
            self.game.render()
            pygame.display.update()

            # Check if Pac-Man's position matches any of the node positions

            # print(pacman_pos)
            valid = self.move_ai(net1, game_info, genome1)
            # print(valid)
            game_info = self.game.loop()
            self.game.render()
            pygame.display.update()
            if game_info is not None:

                valid = self.move_ai(net1, game_info, genome1)
                if valid:
                    self.calculate_fitness(genome1, game_info)
                elif valid is False:
                    break

    def calculate_fitness(self, genome1,  game_info):
        if genome1.fitness == None:

            genome1.fitness = 0
            genome1.fitness += game_info.score / 10

        else:
            genome1.fitness += game_info.score / 10


        #genome2.fitness += game_info.score / 10 + game_info.time_elapsed / 1000
        # print(genome1.fitness)

    def move_ai(self, net, game_info, genome1):
        #print("start of move ai")
        dt = self.clock.tick(300) / 1000.0
        pacman_pos = (game_info.pacman_pos_x.x, game_info.pacman_pos_x.y)
        for node_pos in self.turning_points:
            if pacman_pos == node_pos:

                output1 = net.activate((game_info.pacman_pos_x.x, game_info.pacman_pos_x.y,
                                        game_info.ghosts_pos[0].x - game_info.pacman_pos_x.x,
                                        game_info.ghosts_pos[0].y - game_info.pacman_pos_x.y,
                                        game_info.ghosts_pos[1].x - game_info.pacman_pos_x.x,
                                        game_info.ghosts_pos[1].y - game_info.pacman_pos_x.y,
                                        game_info.ghosts_pos[2].x - game_info.pacman_pos_x.x,
                                        game_info.ghosts_pos[2].y - game_info.pacman_pos_x.y,
                                        game_info.ghosts_pos[3].x - game_info.pacman_pos_x.x,
                                        game_info.ghosts_pos[3].y - game_info.pacman_pos_x.y,
                                        game_info.score
                                        ))
                decision = output1.index(max(output1))
                if decision == 0:
                    # self.game.pacman.direction = UP
                    keyboard.press('w')
                    keyboard.release('a')
                    keyboard.release('s')
                    keyboard.release('d')
                    # print("up")

                elif decision == 1:
                    keyboard.press('s')
                    keyboard.release('d')
                    keyboard.release('a')
                    keyboard.release('w')
                    # print("down")

                elif decision == 2:
                    keyboard.press('a')
                    keyboard.release('d')
                    keyboard.release('s')
                    keyboard.release('w')
                    # print("left")

                elif decision == 3:
                    keyboard.press('d')
                    keyboard.release('s')
                    keyboard.release('a')
                    keyboard.release('w')

                    # print("right")
            else:
                pass
        self.pacman.update(dt)
        pacman_pos = (game_info.pacman_pos_x.x, game_info.pacman_pos_x.y)
        # print(pacman_pos)
        current_position = (game_info.pacman_pos_x.x, game_info.pacman_pos_x.y)
        if game_info.lives < 5:
            if genome1.fitness == None:
                genome1.fitness = 0
                genome1.fitness -= 500
            else:

                genome1.fitness -= 500
            return False

        elif current_position != self.last_position:
            self.last_position = current_position
            current_time = datetime.datetime.now()
            time_string = current_time.strftime("%H%M%S")
            self.last_position_time = time_string
            # print(self.last_position_time)

            # print(last_position)
            # last_position_time = time.time()
        elif current_position == self.last_position:
            current_time = datetime.datetime.now()
            time_string = current_time.strftime("%H%M%S")
            # print(int(time_string) - int(self.last_position_time))
            if int(time_string) - int(self.last_position_time) > 1.5:
                # print('idle')
                if genome1.fitness == None:
                    genome1.fitness = 0
                    genome1.fitness -= 30
                else:

                    genome1.fitness -= 30


                return False
            else:
                return True

                # self.calculate_fitness(genome1, genome2, game_info)
                pass


        #print("end of move ai")




def train_ai(genome1, config):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    #net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

    run = True
    pgame = pacman_game()
    pgame.game.startGame()
    # self.pause.setPause(playerPaused=True)
    pgame.game.pause.setPause(playerPaused=False)
    time_stamp = None
    last_position = None  # Store the last position of the agent
    last_position_time = None

    while run:
        dt = pgame.clock.tick(300) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        pgame.game.pacman.update(dt)
        game_info = pgame.game.loop()
        pgame.game.render()
        pygame.display.update()

        # Check if Pac-Man's position matches any of the node positions

        # print(pacman_pos)
        #valid = pgame.move_ai(net1, game_info, genome1)
        #print("valid value is " ,valid)
        # print(valid)
        game_info = pgame.game.loop()
        pgame.game.render()
        pygame.display.update()
        #if game_info is not None:
        if genome1.fitness is None:

            genome1.fitness = 0
        valid = pgame.move_ai(net1, game_info, genome1)
        if valid:
            calculate_fitness(genome1, game_info)

        elif valid is False:
            break
        elif valid == None:
            pass
    return genome1.fitness

def calculate_fitness( genome1, game_info):
    #print("start of calc")
    if genome1.fitness == None:
        print("zero zero zero \n")
        genome1.fitness = 0
    genome1.fitness += game_info.score / 10 + game_info.time_elapsed / 1000
    #print("end of calc")





#fitness_lock = threading.Lock()
'''
def eval_genomes(genomes, config):
    fitness_values = []
    """
    Run each genome against each other one time to determine the fitness.
    """
    workercount = 0

    for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i / len(genomes) * 100), end=" ")
        #genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
            workercount +=1

    print(workercount)




    with concurrent.futures.ProcessPoolExecutor(max_workers=workercount) as executor:
        futures = []
        for i, (genome_id1, genome1) in enumerate(genomes):
            print(round(i / len(genomes) * 100), end=" ")
            #print(genome1)
            genome1.fitness = 1

            for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
                genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
                futures.append(executor.submit(run_game, genome1, genome2, config))

            for future in concurrent.futures.as_completed(futures):
                fitnesses = future.result()
                genome1, genome2, fitness1, fitness2 = fitnesses
                genome1.fitness += fitness1
                #genome2.fitness += fitness2
            print("look here",genome1.fitness)

    print(genome1.fitness)
    #return [(genome_id, genome.fitness) for genome_id, genome in genomes]
'''

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        game = pacman_game()
        game.train_ai(genome, config)
        return genome.fitness

    '''for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i / len(genomes) * 100), end=" ")
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = pacman_game()
            game.train_ai(genome1, genome2, config)'''





def run_game(genome1, genome2, config):
    game = pacman_game()
    game.train_ai(genome1, config)
    return genome1, genome2, genome1.fitness, genome2.fitness


def test_best_network(config):
    with open("network.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #with open("best.pickle", "rb") as f:
        #genome = pickle.load(f)

        # Convert loaded genome into required data structure
    #genomes = [(1, genome)]
    #print(genomes)

    #pygame.display.set_caption("Pong")
    pac_game = pacman_game()
    pac_game.game.startGame()
    pac_game.game.pause.setPause(playerPaused=False)
    pac_game.test_ai(winner_net)


def run_neat(config_f):
    config = config_f
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-20')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # Create the ParallelEvaluator with multiprocessing
    for genome_id, genome in p.population.items():
        genome.fitness = 0

    evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), train_ai)

    # Run the NEAT algorithm with the evaluator
    winner = p.run(evaluator.evaluate, 5)

    # Save the winner
    with open("network.pickle", "wb") as f:
        pickle.dump(winner, f)
    print("Saved the best genome.")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    #test_best_network(config)