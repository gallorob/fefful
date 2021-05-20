import argparse
import os
from typing import Any, Tuple, Union, List, Dict
from operator import itemgetter
import random

import neat
import numpy as np
import torch as th
from google.auth.transport import grpc

import minecraft_pb2_grpc
from fitness_estimator import ArtifactsBuffer, FitnessEstimatorWrapper
from history_manager import HistoryManager
from minecraft_pb2 import *
from minecraft_pb2_grpc import *
from pytorch_neat.cppn import create_cppn, Node
from pytorch_neat.neat_reporter import LogReporter
from settings import MCSettings


class MCEvaluator:
    def __init__(self,
                 client: Any,
                 mc_settings: MCSettings,
                 additional_args: Dict[str, Any]):
        self.client = client
        self.pop_size = mc_settings.evaluatable_artifacts
        self.mc_settings = mc_settings
        self.buffer = ArtifactsBuffer(settings=mc_settings)
        self.fitness_estimator = FitnessEstimatorWrapper(
            test_threshold=self.mc_settings.test_threshold,
            net_args={
                'in_channels_conv': 2,  # block type and rotation
                'out_channels_conv': self.mc_settings.conv_channels,
                'in_out_channels_res': self.mc_settings.conv_channels,
                'n_features': self.mc_settings.conv_channels * self.mc_settings.artifact_width * self.mc_settings.artifact_height * self.mc_settings.artifact_depth
            }
        )
        # if provided, load the fitness estimator status
        if additional_args.get('timestep') is not None:
            self.fitness_estimator.load(to_resume=additional_args.get('to_resume'),
                                        where=mc_settings.nets_folder,
                                        timestep=additional_args.get('timestep'))

        self.iterations_counter = 0

        self.history_manager = HistoryManager(mc_settings.history_folder)
        if additional_args.get('history') is not None:
            self.history_manager.load(filename=additional_args.get('history'))

        self.generations_counter = 0

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
                max=Point(x=self.mc_settings.x0 + (
                        self.mc_settings.artifact_width + self.mc_settings.artifact_spacing) * self.pop_size,
                          y=self.mc_settings.y0 + self.mc_settings.artifact_height,
                          z=self.mc_settings.z0 + self.mc_settings.artifact_depth)
            ),
            type=AIR
        ))

    def place_signs(self):
        blocks = []
        for i in range(self.pop_size):
            blocks.append(Block(
                position=Point(
                    x=self.mc_settings.x0 + (i * (self.mc_settings.artifact_width + self.mc_settings.artifact_spacing)),
                    y=self.mc_settings.y0,
                    z=self.mc_settings.z0 - 3),
                type=STANDING_SIGN,
                orientation=NORTH
            ))

        self.client.spawnBlocks(Blocks(blocks=blocks))

    def _minimum_criterion(self,
                           artifacts: List[np.ndarray],
                           genomes: List[neat.DefaultGenome]) -> Tuple[List[np.ndarray], List[neat.DefaultGenome]]:
        def is_promising(artifact):
            return (0. <= artifact[0,:,:,:]).all() and np.std(
                artifact[0, :, :, :]) > self.mc_settings.min_block_type_std and np.std(
                artifact[1, :, :, :]) > self.mc_settings.min_block_rot_std and \
                (artifact[0,:,:,:] <= 1./len(self.mc_settings.admissible_blocks)).sum() > self.mc_settings.min_air_fraction*np.prod(artifact.shape[1:])

        promising = [i for i, artifact in enumerate(artifacts) if is_promising(artifact)]
        if len(promising) > self.pop_size:
            print(f'minimum_criterion: too many survivors ({len(promising)}), decimating to {self.pop_size}')
            promising = np.random.choice(promising, size=self.pop_size)
        get_promising = itemgetter(*promising)
        return get_promising(artifacts), get_promising(genomes)

    def _generate_artifacts(self,
                            genomes: List[neat.DefaultGenome],
                            config: neat.Config):
        all_artifacts = []
        for i, (n, genome) in enumerate(genomes):
            block_out, rot_out = self.make_net(genome, config)
            # build artifact in MC
            artifact = np.zeros(shape=(2,
                                       self.mc_settings.artifact_width,
                                       self.mc_settings.artifact_height,
                                       self.mc_settings.artifact_depth))
            for x in range(self.mc_settings.artifact_width):
                for y in range(self.mc_settings.artifact_height):
                    for z in range(self.mc_settings.artifact_depth):
                        block_type = block_out(x_in=th.as_tensor(x / self.mc_settings.artifact_width),
                                               y_in=th.as_tensor(y / self.mc_settings.artifact_height),
                                               z_in=th.as_tensor(z / self.mc_settings.artifact_depth))
                        block_rot = rot_out(x_in=th.as_tensor(x / self.mc_settings.artifact_width),
                                            y_in=th.as_tensor(y / self.mc_settings.artifact_height),
                                            z_in=th.as_tensor(z / self.mc_settings.artifact_depth))

                        artifact[:, x, y, z] = [block_type, block_rot]
            all_artifacts.append(artifact)

        return all_artifacts

    def _generate_blocks(self,
                         artifacts: List[np.ndarray]) -> List[Block]:
        blocks = []
        for i, artifact in enumerate(artifacts):
            for x in range(self.mc_settings.artifact_width):
                for y in range(self.mc_settings.artifact_height):
                    for z in range(self.mc_settings.artifact_depth):
                        block_type, block_rot = artifact[:, x, y, z]
                        # transform block to valid enums
                        block_type = self.mc_settings.val_to_enum(val=block_type,
                                                                  block=True)
                        block_rot = self.mc_settings.val_to_enum(val=block_rot,
                                                                 block=False)
                        block_x = self.mc_settings.x0 + x + (
                                i * (self.mc_settings.artifact_width + self.mc_settings.artifact_spacing))
                        block_y = self.mc_settings.y0 + y
                        block_z = self.mc_settings.z0 + z
                        blocks.append(Block(position=Point(x=block_x,
                                                           y=block_y,
                                                           z=block_z),
                                            type=block_type,
                                            orientation=block_rot)
                                      )
        return blocks

    def eval_genomes(self,
                     genomes: Union[neat.DefaultGenome, Tuple[int, List[neat.DefaultGenome]]],
                     config: neat.Config,
                     debug: bool = False) -> None:
        # used when evaluating best single genome instead of a population of genomes
        if type(genomes) is neat.DefaultGenome:
            print(f"Called `eval_genomes` with single genome; {'expected' if debug else 'unexpected'}.")
            return

        # check if we can use the estimator according to number of generations
        if self.iterations_counter % self.mc_settings.train_interval == 0:
            self.fitness_estimator.can_estimate = False

        # clear user area
        self.clear_space()

        # generate artifacts
        all_artifacts = self._generate_artifacts(genomes=genomes,
                                                 config=config)

        promising_artifacts, promising_genomes = self._minimum_criterion(all_artifacts, genomes)
        print(f'{len(promising_artifacts)} artifacts survived the MC')

        blocks = self._generate_blocks(artifacts=promising_artifacts)

        # User-based fitness assignment
        # spawn blocks on the MC world
        self.client.spawnBlocks(Blocks(blocks=blocks))

        for _, genome in genomes:
            genome.fitness = 0.
        # if possible, assign automatically previously applied choices or fitnesses
        if self.history_manager.has_choices(generation=self.generations_counter):
            human, fitnesses = self.history_manager.get_choices(generation=self.generations_counter)
            for i, (_, genome) in enumerate(promising_genomes):
                if not human:
                    genome.fitness = fitnesses[i]
                else:
                    genome.fitness = 1. if i + 1 in fitnesses else 0.

        elif not self.fitness_estimator.can_estimate:
            # get user's input
            fitnesses = list(map(int, input('Enter interesting artifacts (csv): ').split(',')))
            # sanity checks
            assert len(
                fitnesses) <= self.pop_size, f'Too many fitness values: expected {self.pop_size}, received {len(fitnesses)}'
            assert max(fitnesses) <= self.pop_size, f'Unexpected artifact number: {max(fitnesses)}'
            assert min(fitnesses) > 0, f'Unexpected artifact number: {min(fitnesses)}'
            # add artifacts and fitnesses to the buffer
            for i, artifact in enumerate(promising_artifacts):
                self.buffer.add(artifact=artifact,
                                fitness=1. if i + 1 in fitnesses else 0.)
            # save choices in history
            self.history_manager.add_choices(generation=self.generations_counter,
                                             choices=fitnesses)
            # assign fitness
            for i, (_, genome) in enumerate(promising_genomes):
                genome.fitness = 1. if i + 1 in fitnesses else 0.
        else:
            # get network's estimates
            fitnesses = self.fitness_estimator.estimate(artifacts=promising_artifacts)
            # save fitness in history
            self.history_manager.add_choices(generation=self.generations_counter,
                                             choices=fitnesses.tolist(),
                                             human=False)
            # assign fitness
            for i, (_, genome) in enumerate(promising_genomes):
                genome.fitness = fitnesses[i]

        # train estimator if possible
        if self.buffer.at_capacity:
            if not self.fitness_estimator.can_estimate:
                dataloaders = self.buffer.prepare()
                self.fitness_estimator.train(dataloaders=dataloaders,
                                             epochs=self.mc_settings.train_epochs)
                self.fitness_estimator.save(to_resume=True,
                                            where=self.mc_settings.nets_folder)
            else:
                self.iterations_counter += 1

        self.generations_counter += 1


def run(n_generations: int,
        additional_args: Dict[str, Any]):
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
        mc_settings=mc_settings,
        additional_args=additional_args
    )

    evaluator.place_signs()

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

    evaluator.fitness_estimator.save(to_resume=False,
                                     where=mc_settings.nets_folder)
    evaluator.history_manager.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoCraft Challenge / SAI Project')
    parser.add_argument('--seed', type=int, dest='random_seed', default=12345,
                        help='RNG seed for reproducibility (default: 123456)')
    parser.add_argument('--n', type=int, dest='n_generations', default=10,
                        help='Number of generations to run the evolution for (default: 10)')
    parser.add_argument('--history', type=str, dest='history', default=None,
                        help='Optional; Name of history file of past choices / fitnesses')
    parser.add_argument('--net', type=str, dest='net_name', default=None,
                        help='Optional; Name of Fitness Estimator network')
    parser.add_argument('--resume', dest='to_resume', action='store_true')
    parser.add_argument('--from_scratch', dest='to_resume', action='store_false')
    parser.set_defaults(to_resume=False)

    args = parser.parse_args()

    th.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    # TODO is this needed?
    np.random.seed(args.random_seed)

    run(n_generations=args.n_generations,
        additional_args={
            'to_resume': args.to_resume,
            'timestep': args.net_name,
            'history': args.history
        })
