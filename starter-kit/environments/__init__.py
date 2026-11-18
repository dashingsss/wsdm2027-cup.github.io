"""Dataset selection; fail before executing any tasks if a domain is unavailable."""
from importlib import import_module

DATASETS = ("alfworld", "webshop", "hotpotqa")


def get_environment(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    module = import_module(f"environments.{name}")
    if not module.AVAILABLE:
        raise ValueError(f"Dataset {name} is not available yet: {module.UNAVAILABLE_REASON}")
    return module




# Shared JSON data contract for the two new domains; ALFWorld keeps its original loader.
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from functools import lru_cache

from algorithms.base import PlanningInput


@lru_cache(maxsize=16)
def _read_rows(path, mtime_ns):
    with Path(path).open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row['task_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate task IDs in {path}')
    return {row['task_id']: row for row in rows}


def task_rows(root):
    path = Path(root).resolve() / 'tasks.jsonl'
    if not path.is_file():
        raise ValueError(f'Missing {path}. Prepare this domain data first.')
    return _read_rows(str(path), path.stat().st_mtime_ns)






def json_task_type(task_id):
    return task_id.split('/')[2]




def json_fingerprints(root, tasks):
    rows = task_rows(root)
    manifest = (Path(root) / 'manifest.json').read_bytes()
    return {task: hashlib.sha256(manifest + json.dumps(rows[task], sort_keys=True).encode()).hexdigest()
            for task in tasks}


class TextAdapter:
    """Task bookkeeping only. Subclasses implement start() and perform(action)."""
    def __init__(self, data_root, task_id, max_actions=50):
        self.root = Path(data_root).resolve()
        if max_actions < 1: raise ValueError('max_actions must be positive')
        self.task_id = task_id
        try: self.task = copy.deepcopy(task_rows(self.root)[task_id])
        except KeyError: raise ValueError(f'Unknown task: {task_id}') from None
        self.max_actions = max_actions
        self.goal = self.task['goal']
        self.observation = ''
        self.available_actions = []
        self.history = []
        self.steps = self.attempts = 0
        self.done = self.success = False
        self.score = 0.0
        self.submitted = False

    def reset(self):
        self.close()
        self.history = []
        self.steps = self.attempts = 0
        self.done = self.success = self.submitted = False
        self.score = 0.0
        self.start()
        return self.snapshot()

    def snapshot(self):
        # Never expose self.task: it can contain evaluator-only labels.
        return {'observation': self.observation, 'available_actions': list(self.available_actions),
                'done': self.done, 'success': self.success, 'score': self.score,
                'environment_steps': self.steps, 'action_attempts': self.attempts}

    def planning_input(self):
        return PlanningInput(self.task_id, self.goal, self.observation,
                             tuple(self.available_actions), tuple(copy.deepcopy(self.history)), self.domain)

    def step(self, action):
        if self.done: return {**self.snapshot(), 'error': 'Episode has finished'}
        self.attempts += 1
        before, error, valid = self.observation, None, True
        try:
            if not isinstance(action, str) or len(action) > 8000:
                raise ValueError('Action must be a string of at most 8000 characters')
            self.perform(action)
            self.steps += 1
        except ValueError as exc:
            error, valid = str(exc), False
        self.done = self.done or self.attempts >= self.max_actions
        result = self.snapshot()
        if error: result['error'] = error
        self.history.append({'action': action, 'valid': valid, 'before': before, **result})
        return result

    def close(self):
        pass


def java_home():
    homes = sorted((Path(sys.prefix) / 'lib/jvm').glob('*/Contents/Home'))
    if homes:
        os.environ['JAVA_HOME'] = str(homes[-1])
    if not os.environ.get('JAVA_HOME'):
        raise ValueError('JAVA_HOME is missing; install the documented x86_64 JDK 11 in this conda environment')
    return os.environ['JAVA_HOME']


@lru_cache(maxsize=1)
def sparse_search_class():
    """Load unchanged Pyserini 0.17 BM25 code without eager neural/FAISS imports.

    Upstream package __init__ imports optional torch encoders even for Lucene BM25.
    Only package initialization is narrowed; ranking/index classes and JVM are official.
    """
    import importlib
    import importlib.metadata
    import types
    java_home()
    if importlib.metadata.version('pyserini') != '0.17.0':
        raise ValueError('Sparse compatibility loader is verified only for pyserini==0.17.0')
    if 'pyserini.search' in sys.modules:
        return importlib.import_module('pyserini.search.lucene').LuceneSearcher
    import pyserini
    base = Path(pyserini.__file__).parent
    names = ('pyserini.search', 'pyserini.search.lucene')
    try:
        for name, folder in zip(names, (base / 'search', base / 'search/lucene')):
            module = types.ModuleType(name)
            module.__path__ = [str(folder)]
            sys.modules[name] = module
        module = importlib.import_module('pyserini.search._base')
        for key in dir(module):
            if not key.startswith('_'): setattr(sys.modules[names[0]], key, getattr(module, key))
        cls = importlib.import_module('pyserini.search.lucene._searcher').LuceneSearcher
        sys.modules[names[1]].LuceneSearcher = cls
        return cls
    except Exception:
        for name in list(sys.modules):
            if name.startswith('pyserini.search'): sys.modules.pop(name, None)
        raise



def load_trajectories(root):
    """Read public trajectories; provenance and GT are organizer-only records."""
    path = Path(root) / 'trajectories.jsonl'
    rows = list(_read_rows(str(path.resolve()), path.stat().st_mtime_ns).values())
    if not rows:
        raise ValueError('The trajectory library is empty')
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('task_id', 'goal')):
            raise ValueError('Trajectory requires task_id and goal strings')
        if not isinstance(row.get('initial_observation'), str) or not isinstance(row.get('steps'), list) or not row['steps']:
            raise ValueError('Trajectory requires an initial observation and steps')
        for step in row['steps']:
            if not isinstance(step, dict) or not isinstance(step.get('action'), str) or not step['action'].strip() or not isinstance(step.get('observation'), str):
                raise ValueError('Each step requires an action and observation')
    result = [{**r, 'actions': [s['action'] for s in r['steps']],
               'observations': [r['initial_observation']] + [s['observation'] for s in r['steps']]} for r in rows]
    return result, {'accepted': len(result), 'corpus_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
