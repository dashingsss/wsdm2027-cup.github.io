"""Implement your method here; add dependencies to docker/requirements.algorithm.txt.

Build the algorithm image, then use docker/launch.py to run development tasks.
Only the algorithm container imports this module; do not import environments.
"""
from algorithms.base import Guidance, PlanningAlgorithm

class Example(PlanningAlgorithm):
    requires_corpus = False
    def reset(self, initial, corpus):
        self.task_id = initial.task_id
    def get_guidance(self, state):
        return Guidance('Check the current observations before choosing the next action.')

def create_algorithm(config):
    return Example()
