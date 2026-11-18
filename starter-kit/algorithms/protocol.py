"""Versioned JSON contract; no environment modules or executable payloads."""
from dataclasses import asdict
from .base import PlanningInput, Guidance

VERSION = 1
# Transport safety bound, not a model-context or reference-count budget.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
HISTORY_FIELDS = {
    'action', 'valid', 'before', 'observation', 'available_actions', 'done',
    'success', 'score', 'environment_steps', 'action_attempts', 'error',
    'search_available', 'action_instructions',
}


def state_to_json(state):
    value = asdict(state)
    value['history'] = [{k: v for k, v in row.items() if k in HISTORY_FIELDS}
                        for row in state.history]
    return value


def state_from_json(value):
    if not isinstance(value, dict) or set(value) != {
        'task_id', 'goal', 'observation', 'available_actions', 'history', 'domain'
    }:
        raise ValueError('Unexpected planning-state fields')
    if not all(isinstance(value[k], str) for k in ('task_id', 'goal', 'observation', 'domain')):
        raise ValueError('State text fields must be strings')
    if not isinstance(value['available_actions'], list) or not all(
        isinstance(x, str) for x in value['available_actions']
    ) or not isinstance(value['history'], list):
        raise ValueError('Invalid actions or history')
    if any(not isinstance(row, dict) or set(row) - HISTORY_FIELDS for row in value['history']):
        raise ValueError('Unexpected history fields')
    return PlanningInput(**{**value, 'available_actions': tuple(value['available_actions']),
                            'history': tuple(value['history'])})


def guidance_to_json(value):
    if not isinstance(value, Guidance):
        raise ValueError('Return Guidance, not an action or Python object')
    return {'text': value.text, 'retrieved_ids': list(value.retrieved_ids)}


def guidance_from_json(value):
    if not isinstance(value, dict) or set(value) != {'text', 'retrieved_ids'}:
        raise ValueError('Unexpected guidance fields')
    if not isinstance(value['text'], str) or not isinstance(value['retrieved_ids'], list) or not all(
        isinstance(x, str) for x in value['retrieved_ids']
    ):
        raise ValueError('Invalid guidance types')
    return Guidance(value['text'], tuple(value['retrieved_ids']))
