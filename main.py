import numpy as np
import grpc

import os, sys
import numpy as np
import torch as th
import minecraft_pb2_grpc
from minecraft_pb2 import *
import neat

from pytorch_neat.cppn import create_cppn
from pytorch_neat.neat_reporter import LogReporter


MC_PROPERTIES = {
    'x_start': 0,
    'y_start': 4,
    'z_start': 0,
    'artifact_width': 5,
    'artifact_depth': 5,
    'artifact_height': 5,
    'offset': 2
}

MC_ROTATIONS = [NORTH,
                WEST,
                SOUTH,
                EAST,
                UP,
                DOWN]
MC_BLOCKS = [ACACIA_DOOR,
             ACACIA_FENCE,
             ACACIA_STAIRS,
             AIR,
             BOOKSHELF,
             CHEST,
             COBBLESTONE_WALL,
             DIRT,
             GLASS_PANE,
             GLOWSTONE,
             GRAVEL]


class MCEvaluator:
    def __init__(self, make_net, activate_net, client):
        self.make_net = make_net
        self.activate_net = activate_net
        self.client = client

    def eval_genomes(self, genomes, config, debug=False):
        # used when evaluating best single genome instead of a population of genomes
        if type(genomes) is neat.DefaultGenome:
            return 1.
        # clear user area
        clear_space(self.client,
                    len(genomes))
        # generate artifacts
        all_artifacts = []
        for i, (n, genome) in enumerate(genomes):
            block_out, rot_out = self.make_net(genome, config, 1)
            # build artifact in MC
            artifact = np.zeros(shape=(MC_PROPERTIES.get('artifact_width'),
                                       MC_PROPERTIES.get('artifact_depth'),
                                       MC_PROPERTIES.get('artifact_height'),
                                       2))
            blocks = []
            for x in range(MC_PROPERTIES.get('artifact_width')):
                for y in range(MC_PROPERTIES.get('artifact_depth')):
                    for z in range(MC_PROPERTIES.get('artifact_height')):
                        block_type = block_out(x_in=th.as_tensor(x),
                                               y_in=th.as_tensor(y),
                                               z_in=th.as_tensor(z))
                        block_rot = rot_out(x_in=th.as_tensor(x),
                                            y_in=th.as_tensor(y),
                                            z_in=th.as_tensor(z))

                        artifact[x, y, z, :] = [block_type, block_rot]
                        # transform block to valid enums
                        block_type = MC_BLOCKS[int(block_type * len(MC_BLOCKS)) - 1]
                        block_rot = MC_ROTATIONS[int(block_rot * len(MC_ROTATIONS)) - 1]
                        blocks.append(
                            Block(position=Point(x=MC_PROPERTIES.get('x_start') + x + (i * (MC_PROPERTIES.get('artifact_width') + MC_PROPERTIES.get('offset'))),
                                                 y=MC_PROPERTIES.get('y_start') + y,
                                                 z=MC_PROPERTIES.get('z_start') + z),
                                  type=block_type,
                                  orientation=block_rot)
                        )
            # add to database and spawn artifact in-game
            all_artifacts.append(artifact)
            self.client.spawnBlocks(Blocks(blocks=blocks))
        # get user's input or nn's
        fitnesses = list(map(int, input('Enter interesting artifacts (csv): ').split(',')))
        # assign fitness
        for i, genome in genomes:
            genome.fitness = 1. if i in fitnesses else 0.


def clear_space(client, n_artifacts):
    x0 = MC_PROPERTIES.get('x_start', 0)
    y0 = MC_PROPERTIES.get('y_start', 0)
    z0 = MC_PROPERTIES.get('z_start', 0)

    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=x0,
                      y=y0,
                      z=z0),
            max=Point(x=x0 + (MC_PROPERTIES.get('artifact_width') + MC_PROPERTIES.get('offset')) * n_artifacts,
                      y=y0 + MC_PROPERTIES.get('artifact_depth'),
                      z=z0 + MC_PROPERTIES.get('artifact_height'))
        ),
        type=AIR
    ))

    # spawn arrow on the ground to show direction of artifacts are spawned in
    client.spawnBlocks(
        Blocks(blocks=[
            Block(position=Point(x=x0 - 5,
                                 y=y0 + 1,
                                 z=z0 - 4),
                  type=COBBLESTONE,
                  orientation=NORTH),
            Block(position=Point(x=x0 - 4,
                                 y=y0 + 1,
                                 z=z0 - 4),
                  type=COBBLESTONE,
                  orientation=NORTH),
            Block(position=Point(x=x0 - 3,
                                 y=y0 + 1,
                                 z=z0 - 4),
                  type=COBBLESTONE,
                  orientation=NORTH),
            Block(position=Point(x=x0 - 2,
                                 y=y0 + 1,
                                 z=z0 - 4),
                  type=COBBLESTONE,
                  orientation=NORTH),
            Block(position=Point(x=x0 - 3,
                                 y=y0,
                                 z=z0 - 4),
                  type=COBBLESTONE,
                  orientation=NORTH),
            Block(position=Point(x=x0 - 3,
                                 y=y0 + 2,
                                 z=z0 - 4),
                  type=COBBLESTONE,
                  orientation=NORTH)
        ])
    )


def make_net(genome, config, bs):
    return create_cppn(
        genome,
        config,
        ["x_in", "y_in", 'z_in'],
        ["block_out", 'rotation_out'],
        )


def activate_net(net, states):
    return net.activate(states).numpy()


def run(n_generations):
    channel = grpc.insecure_channel('localhost:5001')
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MCEvaluator(
        make_net, activate_net, client
    )

    def eval_genomes(genomes, config):
        evaluator.eval_genomes(genomes, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("neat.log", evaluator.eval_genomes)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == '__main__':
    run(n_generations=2)
