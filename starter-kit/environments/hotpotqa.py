"""HotpotQA search/read/finish environment with the unmodified official answer EM."""
from environments import json_task_type as task_type
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
from functools import lru_cache

from environments import TextAdapter, json_fingerprints

AVAILABLE = True
DOMAIN = 'hotpotqa'
RUNTIME_PACKAGES = ('openjiuwen', 'rank-bm25', 'ujson')
SYSTEM_PROMPT = '''Answer the question using the fixed document search environment.
Call hotpotqa_step with one complete action string at a time:
- search: put actual entity names or keywords from the question or observed evidence
  inside the brackets, for example search[Marie Curie]. You may search again at any time.
- read: copy a complete read command from available_actions. Its ID comes from an
  actual search result; do not invent an ID or use an article title as an ID.
- finish: put your actual concise answer inside the brackets, for example
  finish[Warsaw]. This submits the answer and ends the task; do not add an explanation.
These examples illustrate syntax, not answers to the current question. Words such
as query, document_id and answer describe parameters, not ready-to-execute actions.
available_actions lists discovered read commands only, not every possible search
or answer. An empty list means search first, not that you cannot act.
Read relevant evidence and follow intermediate entities to solve bridge questions;
for comparisons, find evidence for both subjects. If search returns no useful result,
try a specific entity or different keywords. Retrieved historical examples are advice,
not evidence for this question. Do not claim to have searched or read without a tool call.'''


def raw_data_root(data_root):
    root = Path(data_root).resolve()
    if root.name == 'hotpotqa' and root.parent.name == 'data':
        resources = root.parent.parent / 'resources/hotpotqa'
        if (resources / 'manifest.json').is_file():
            return resources
    return root




def task_fingerprints(root, tasks):
    return json_fingerprints(raw_data_root(root), tasks)


def load_expert_corpus(root):
    from environments import load_trajectories
    return load_trajectories(root)


@lru_cache(maxsize=1)
def official_evaluator():
    path = Path(__file__).resolve().parents[1] / 'third_party/hotpotqa/hotpot_evaluate_v1.py'
    if not path.is_file(): raise ValueError(f'Missing official evaluator: {path}')
    spec = importlib.util.spec_from_file_location('hotpot_official_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(root):
    root = raw_data_root(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not (root / 'documents.sqlite').is_file(): raise ValueError('Missing HotpotQA document index')
    official_evaluator()
    with sqlite3.connect((root / 'documents.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('SELECT rowid FROM documents_fts LIMIT 1').fetchone()
    return {'corpus_scope': manifest['corpus_scope'], 'documents': manifest['documents'],
            'official_fullwiki': False}


class Adapter(TextAdapter):
    domain = DOMAIN
    system_prompt = SYSTEM_PROMPT
    tool_name = 'hotpotqa_step'
    tool_description = 'Search documents, read a returned document ID, or submit the final answer.'
    tool_action_description = (
        'One complete action string. For search, supply actual keywords, e.g. '
        'search[Marie Curie]. For read, copy a command from available_actions. '
        'For finish, supply your actual short answer, e.g. finish[Warsaw]. '
        'Replace the example content with content relevant to the current question; '
        'never send the literal parameter placeholders query, document_id or answer.'
    )

    def __init__(self, data_root, task_id, max_actions=50):
        super().__init__(raw_data_root(data_root), task_id, max_actions)

    def start(self):
        self.db = sqlite3.connect((self.root / 'documents.sqlite').as_uri() + '?mode=ro', uri=True)
        self.visible_ids = set()
        self.available_actions = []
        self.observation = 'Use search to find evidence, read to inspect documents, and finish to submit your answer.'

    def snapshot(self):
        state = super().snapshot()
        if self.done:
            state['available_actions'] = []
        state['action_instructions'] = (
            'Task finished.' if self.done else
            'Search using actual keywords inside search[...]. Read using a complete '
            'command from available_actions; these IDs were returned by earlier searches. '
            'Submit your actual short answer inside finish[...]. Search and finish '
            'are free-text actions, so they are not enumerated in available_actions. '
            'Do not copy the dots or parameter placeholders as content.'
        )
        return state

    def perform(self, action):
        match = re.fullmatch(r'(search|read|finish)\[(.*)\]', action.strip(), re.S)
        if not match: raise ValueError('Use search with actual keywords, read with a returned ID, or finish with your actual answer, each enclosed in brackets.')
        operation, value = match.groups()
        value = value.strip()
        if not value: raise ValueError('Action argument cannot be empty')
        if operation == 'search':
            tokens = re.findall(r'\w+', value.lower(), re.UNICODE)[:40]
            stop = {'the','a','an','of','in','on','is','are','was','were','and','or','to','which','what','who','did','does','do'}
            tokens = list(dict.fromkeys(t for t in tokens if t not in stop)) or tokens
            if not tokens: raise ValueError('Search query has no searchable words')
            query = ' OR '.join('"' + t.replace('"','""') + '"' for t in tokens)
            rows = self.db.execute('''SELECT d.doc_id,d.title,snippet(documents_fts,1,'','', ' … ',48)
                FROM documents_fts JOIN documents d ON d.rowid=documents_fts.rowid
                WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts,5.0,1.0),d.doc_id LIMIT 10''', (query,)).fetchall()
            self.visible_ids.update(row[0] for row in rows)
            self.available_actions = [f'read[{doc_id}]' for doc_id in sorted(self.visible_ids)]
            self.observation = json.dumps([{'id': x[0], 'title': x[1], 'snippet': x[2]} for x in rows], ensure_ascii=False)
        elif operation == 'read':
            if value not in self.visible_ids:
                raise ValueError('This ID has not been returned by search. Copy a read command from available_actions; if it is empty, search with actual keywords first.')
            row = self.db.execute('SELECT title,text FROM documents WHERE doc_id=?', (value,)).fetchone()
            self.observation = json.dumps({'id':value,'title':row[0],'text':row[1]}, ensure_ascii=False)
        else:
            self.score = float(official_evaluator().exact_match_score(value, self.task['answer']))
            self.success = self.score == 1
            self.done = self.submitted = True
            self.available_actions = []
            self.observation = 'Answer submitted. Episode finished.'

    def close(self):
        if getattr(self, 'db', None) is not None:
            self.db.close()
            self.db = None




