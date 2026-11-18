"""Algorithm-container loader. Never call this from the official evaluator."""
from copy import deepcopy
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys
from uuid import uuid4
from .base import PlanningAlgorithm


def create_algorithm(path, *, options=None, seed=42, domain='alfworld'):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != '.py':
        raise ValueError('Method must be a Python file')
    name = 'participant_' + uuid4().hex
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    factory = getattr(module, 'create_algorithm', None)
    if not callable(factory):
        raise ValueError('Method must export create_algorithm(config)')
    algorithm = factory({'seed': seed, 'domain': domain, 'options': deepcopy(options or {})})
    if not isinstance(algorithm, PlanningAlgorithm) or type(algorithm.requires_corpus) is not bool:
        raise ValueError('Factory must return a PlanningAlgorithm with boolean requires_corpus')
    return algorithm
