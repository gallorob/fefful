import os
import configparser
from typing import Any, Tuple, Union, List

import neat
import numpy as np
import torch as th
from google.auth.transport import grpc

import minecraft_pb2_grpc
from minecraft_pb2_grpc import *
from minecraft_pb2 import *
from pytorch_neat.cppn import create_cppn, Node
from pytorch_neat.neat_reporter import LogReporter


class MCSettings:
    def __init__(self,
                 filename: str):
        config = configparser.ConfigParser()
        config.read(filename)
        self.start_coordinates = [
            config['START POSITION'].getint('x'),
            config['START POSITION'].getint('y'),
            config['START POSITION'].getint('z')
        ]
        self.artifact_dimensions = [
            config['ARTIFACT'].getint('width'),
            config['ARTIFACT'].getint('height'),
            config['ARTIFACT'].getint('depth')
        ]
        self.artifact_spacing = config['ARTIFACT'].getint('spacing')
        self.admissible_rotations = config['ADMISSIBILES'].get('rotations').replace(' ', '').upper().split(',')
        self.admissible_blocks = config['ADMISSIBILES'].get('blocks').replace(' ', '').upper().split(',')

        self.buffer_capacity = 40

    @property
    def x0(self):
        return self.start_coordinates[0]

    @property
    def y0(self):
        return self.start_coordinates[1]

    @property
    def z0(self):
        return self.start_coordinates[2]

    @property
    def artifact_width(self):
        return self.artifact_dimensions[0]

    @property
    def artifact_depth(self):
        return self.artifact_dimensions[2]

    @property
    def artifact_height(self):
        return self.artifact_dimensions[1]

    def val_to_enum(self,
                    val: float,
                    block: bool = True):
        l = self.admissible_blocks if block else self.admissible_rotations
        e = BlockType if block else Orientation
        return e.Value(l[int(val * len(l)) - 1])  # assumes val <= 1


class MCEvaluator:
    def __init__(self,
                 client: Any,
                 pop_size: int,
                 mc_settings: MCSettings):
        self.client = client
        self.pop_size = pop_size
        self.mc_settings = mc_settings

    @staticmethod
    def make_net(genome: neat.DefaultGenome,
                 config: neat.Config) -> Tuple[Node, Node]:
        return create_cppn(
            genome,
            config,
            ["x_in", "y_in", 'z_in'],
            ["block_out", 'rotation_out'],
        )

    def clear_space(self):
        self.client.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=self.mc_settings.x0,
                          y=self.mc_settings.y0,
                          z=self.mc_settings.z0),
                max=Point(x=self.mc_settings.x0 + (self.mc_settings.artifact_width + self.mc_settings.artifact_spacing) * self.pop_size,
                          y=self.mc_settings.y0 + self.mc_settings.artifact_height,
                          z=self.mc_settings.z0 + self.mc_settings.artifact_depth)
            ),
            type=AIR
        ))

        # spawn arrow on the ground to show direction of artifacts are spawned in
        self.client.spawnBlocks(
            Blocks(blocks=[
                Block(position=Point(x=self.mc_settings.x0 - 5,
                                     y=self.mc_settings.y0 + 1,
                                     z=self.mc_settings.z0 - 4),
                      type=COBBLESTONE,
                      orientation=NORTH),
                Block(position=Point(x=self.mc_settings.x0 - 4,
                                     y=self.mc_settings.y0 + 1,
                                     z=self.mc_settings.z0 - 4),
                      type=COBBLESTONE,
                      orientation=NORTH),
                Block(position=Point(x=self.mc_settings.x0 - 3,
                                     y=self.mc_settings.y0 + 1,
                                     z=self.mc_settings.z0 - 4),
                      type=COBBLESTONE,
                      orientation=NORTH),
                Block(position=Point(x=self.mc_settings.x0 - 2,
                                     y=self.mc_settings.y0 + 1,
                                     z=self.mc_settings.z0 - 4),
                      type=COBBLESTONE,
                      orientation=NORTH),
                Block(position=Point(x=self.mc_settings.x0 - 3,
                                     y=self.mc_settings.y0,
                                     z=self.mc_settings.z0 - 4),
                      type=COBBLESTONE,
                      orientation=NORTH),
                Block(position=Point(x=self.mc_settings.x0 - 3,
                                     y=self.mc_settings.y0 + 2,
                                     z=self.mc_settings.z0 - 4),
                      type=COBBLESTONE,
                      orientation=NORTH)
            ])
        )

    def minimum_criterion(self, artifact):
        return True

    def eval_genomes(self,
                     genomes: Union[neat.DefaultGenome, List[neat.DefaultGenome]],
                     config: neat.Config,
                     debug: bool = False) -> None:
        # used when evaluating best single genome instead of a population of genomes
        if type(genomes) is neat.DefaultGenome:
            print(f"Called `eval_genomes` with single genome; {'expected' if debug else 'unexpected'}.")
            return
        # clear user area
        self.clear_space()
        # generate artifacts
        all_artifacts = []
        for i, (n, genome) in enumerate(genomes):
            block_out, rot_out = self.make_net(genome, config)
            # build artifact in MC
            artifact = np.zeros(shape=(self.mc_settings.artifact_width,
                                       self.mc_settings.artifact_height,
                                       self.mc_settings.artifact_depth,
                                       2))
            for x in range(self.mc_settings.artifact_width):
                for y in range(self.mc_settings.artifact_height):
                    for z in range(self.mc_settings.artifact_depth):
                        block_type = block_out(x_in=th.as_tensor(x / self.mc_settings.artifact_width),
                                               y_in=th.as_tensor(y / self.mc_settings.artifact_height),
                                               z_in=th.as_tensor(z / self.mc_settings.artifact_depth))
                        block_rot = rot_out(x_in=th.as_tensor(x / self.mc_settings.artifact_width),
                                            y_in=th.as_tensor(y / self.mc_settings.artifact_height),
                                            z_in=th.as_tensor(z / self.mc_settings.artifact_depth))

                        artifact[x, y, z, :] = [block_type, block_rot]
            # add to database
            all_artifacts.append(artifact)

        # filter artifacts with the Minimum Criterion and spawn survivors in MC
        decent_artifacts = [artifact for artifact in all_artifacts if self.minimum_criterion(artifact)]
        print(f'{len(decent_artifacts)} artifacts survived the minimum criterion')
        for i, artifact in enumerate(decent_artifacts):
            blocks = []
            for x in range(self.mc_settings.artifact_width):
                for y in range(self.mc_settings.artifact_height):
                    for z in range(self.mc_settings.artifact_depth):
                        block_type, block_rot = artifact[x, y, z, :]
                        # transform block to valid enums
                        block_type = self.mc_settings.val_to_enum(val=block_type,
                                                                  block=True)
                        block_rot = self.mc_settings.val_to_enum(val=block_rot,
                                                                 block=False)
                        block_x = self.mc_settings.x0 + x + \
                                    (i * (self.mc_settings.artifact_width + \
                                     self.mc_settings.artifact_spacing))
                        block_y = self.mc_settings.y0 + y
                        block_z = self.mc_settings.z0 + z
                        blocks.append(
                            Block(position=Point(x=block_x, y=block_y, z=block_z),
                                  type=block_type,
                                  orientation=block_rot)
                        )

            self.client.spawnBlocks(Blocks(blocks=blocks))

        # get user's input
        fitnesses = list(map(int, input('Enter interesting artifacts (csv): ').split(',')))
        assert len(fitnesses) <= self.pop_size, f'Too many fitness values: expected {self.pop_size}, received {len(fitnesses)}'
        assert max(fitnesses) <= self.pop_size, f'Unexpected artifact number: {max(fitnesses)}'
        assert min(fitnesses) > 0, f'Unexpected artifact number: {min(fitnesses)}'
        # assign fitness
        for i, genome in genomes:
            genome.fitness = 1. if i + 1 in fitnesses else 0.


def run(n_generations: int):
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

    mc_settings = MCSettings(filename=os.path.join(os.path.dirname(__file__), "experiment.cfg"))

    evaluator = MCEvaluator(
        client=client,
        pop_size=config.pop_size,
        mc_settings=mc_settings
    )

    def eval_genomes(genomes: Union[neat.DefaultGenome, List[neat.DefaultGenome]],
                     config: neat.Config) -> None:
        evaluator.eval_genomes(genomes, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("neat.log", evaluator.eval_genomes)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)

    # saving? resuming? Perhaps pickle?


if __name__ == '__main__':
    run(n_generations=10)
