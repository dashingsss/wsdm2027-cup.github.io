"""Task-stable random reference, sampled once and reused throughout the episode."""
import hashlib
import json
import random

from algorithms.base import Guidance, PlanningAlgorithm


ADVICE = ('Follow the current task goal and real observations. An available action is not necessarily '
          'useful; avoid undoing requirements already satisfied. Historical examples, if supplied, '
          'are advice only: adapt their objects, IDs and actions to the current task. '
          'Their observations are not evidence obtained in the current task.')

def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def validate_corpus(corpus):
    if not corpus:
        raise ValueError('Retrieval needs a nonempty trajectory corpus')
    ids = set()
    for row in corpus:
        for field in ('task_id', 'goal'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Trajectory needs a nonempty {field}')
        if row['task_id'] in ids:
            raise ValueError('Duplicate trajectory task_id')
        ids.add(row['task_id'])
        actions = row.get('actions')
        if not isinstance(actions, (list, tuple)) or not actions or any(
                not isinstance(a, str) or not a.strip() for a in actions):
            raise ValueError('Trajectory needs nonempty string actions')

        observations = row.get('observations')
        if (not isinstance(observations, (list, tuple)) or not observations
                or not isinstance(observations[0], str)):
            raise ValueError('Trajectory needs a string initial observation')
        if len(observations) != len(actions) + 1 or any(not isinstance(o, str) for o in observations):
            raise ValueError('Trajectory needs one initial observation and one return per action')


def candidates(corpus, state):
    result = [i for i, row in enumerate(corpus)
              if row['task_id'] != state.task_id]
    if not result:
        raise ValueError('No reference trajectories other than this task')
    return sorted(result, key=lambda i: corpus[i]['task_id'])


def format_guidance(rows, max_chars=8000):
    """Bounded historical action/return pairs; never current-task evidence."""
    validate_corpus(rows)
    allowance = (max_chars - len(ADVICE) - 2 * len(rows)) // max(1, len(rows))
    if allowance < 600:
        raise ValueError('Guidance budget too small for top_k; reduce top_k')

    def clip(text, limit):
        marker = ' [truncated]'
        return text if len(text) <= limit else text[:limit - len(marker)] + marker

    sections = [ADVICE]
    for row in rows:
        header = (f"Historical example goal: {clip(row['goal'], allowance // 5)}\n"
                  f"Initial observation: {clip(row['observations'][0], allowance // 4)}\n")
        omitted = '[Remaining historical steps omitted.]'
        budget = allowance - len(header) - len(omitted)
        step_allowance = budget // len(row['actions'])
        shown = []
        for number, (action, observation) in enumerate(zip(row['actions'], row['observations'][1:]), 1):
            label = f'{number}. Action: {action}\n   Historical return: '
            return_limit = max(64, min(1000, step_allowance - len(label) - 1))
            pair = label + clip(observation, return_limit) + '\n'
            if len(pair) > budget:
                break
            shown.append(pair)
            budget -= len(pair)
        if not shown:
            raise ValueError('Guidance budget cannot fit one complete action/return pair')
        tail = omitted if len(shown) < len(row['actions']) else ''
        sections.append(header + ''.join(shown) + tail)
    return Guidance('\n\n'.join(sections), tuple(row['task_id'] for row in rows))


class Random(PlanningAlgorithm):
    def __init__(self, top_k=1, seed=42, max_guidance_chars=8000):
        self.top_k = positive_int(top_k, 'top_k')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        self.seed = seed
        self.max_guidance_chars = positive_int(max_guidance_chars, 'max_guidance_chars')
        self.corpus = None
        self.guidance = Guidance()

    def prepare(self, corpus):
        validate_corpus(corpus)
        self.corpus = corpus

    def reset(self, initial, corpus):
        self.guidance = Guidance()
        if self.corpus is not corpus:
            self.prepare(corpus)
        eligible = candidates(corpus, initial)
        key = json.dumps([self.seed, initial.task_id], ensure_ascii=False)
        rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big'))
        selected = rng.sample(eligible, min(self.top_k, len(eligible)))
        self.guidance = format_guidance([corpus[i] for i in selected], self.max_guidance_chars)

    def get_guidance(self, state):
        return self.guidance


def create_algorithm(config):
    return Random(top_k=config.get('top_k', 1), seed=config.get('seed', 42),
                  max_guidance_chars=config.get('options', {}).get('max_guidance_chars', 8000))
