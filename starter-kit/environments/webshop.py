"""Official WebShop text engine with a complete, disk-backed product catalog."""
from environments import json_task_type as task_type
import copy
import sqlite3
from collections.abc import Mapping, Sequence
import importlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from functools import lru_cache

from environments import TextAdapter, json_fingerprints, sparse_search_class, _read_rows

AVAILABLE = True
DOMAIN = 'webshop'
RUNTIME_PACKAGES = ('openjiuwen', 'pyserini', 'spacy', 'en-core-web-sm', 'gym', 'thefuzz')
SOURCE_COMMIT = '64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd'
SYSTEM_PROMPT = """Complete the shopping instruction using the webshop_step tool, one action at a time.
Read the current page and decide what to do next. To search, write your own actual
search words inside brackets, for example search[black running shoes]. The words
must come from your current task; the example is only a syntax illustration.
Search is allowed when search_available is true. available_actions contains only
complete click actions that may be copied exactly. For example click[buy now]
submits the purchase and ends the task. Check the product description, price and
requested options before buying. If search is unavailable, click[back to search]
when that button is listed. Never send placeholder words such as keywords or query.
An option mentioned in a product title is not automatically selected. On the product
page, explicitly click each requested selectable option (such as size, color or pack)
and check the returned selection before buying. Clicking buy now is final: it ends
the episode even when the purchased product only partially satisfies the task.
Use observations to check progress. Historical trajectories, if supplied, are advice;
verify the current product and options rather than copying old product IDs.
Success and partial reward are determined by the environment.
Do not claim to have performed an action without calling the tool.
"""


def raw_data_root(data_root):
    root = Path(data_root).resolve()
    if root.name == 'webshop' and root.parent.name == 'data':
        resources = root.parent.parent / 'resources/webshop'
        if (resources/'manifest.json').is_file():
            return resources
    return root


def demonstration_tasks(root):
    path = raw_data_root(root)/'curation/demonstration_tasks.jsonl'
    return _read_rows(str(path), path.stat().st_mtime_ns) if path.exists() else {}




def task_fingerprints(root, tasks):
    return json_fingerprints(raw_data_root(root), tasks)


def load_expert_corpus(root):
    from environments import load_trajectories
    return load_trajectories(root)


def prepare_source():
    """Validate prepared text-mode sources without writing runtime files."""
    source = Path(__file__).resolve().parents[1] / 'third_party/webshop'
    if not (source/'web_agent_site').is_dir():
        raise ValueError('Build the official image to install the pinned WebShop sources')
    if (source / '.git').exists():
        actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
        if actual != SOURCE_COMMIT: raise ValueError('Unexpected WebShop source commit')
    else:
        # Review ZIPs omit Git history; verify the packaged source files instead.
        import hashlib
        manifest_path = source / 'SOURCE_MANIFEST.json'
        if not manifest_path.is_file():
            raise ValueError('Missing WebShop source manifest; install the complete source bundle')
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get('files', {})
        required = {'web_agent_site/engine/engine.py', 'web_agent_site/envs/web_agent_text_env.py',
                    'web_agent_site/envs/__init__.py', 'web_agent_site/utils.py'}
        if manifest.get('commit') != SOURCE_COMMIT or not required.issubset(files):
            raise ValueError('Unexpected WebShop source manifest')
        for name, expected in files.items():
            member = (source / name).resolve()
            if not member.is_relative_to(source.resolve()) or not member.is_file():
                raise ValueError('Invalid file in WebShop source manifest')
            if hashlib.sha256(member.read_bytes()).hexdigest() != expected:
                raise ValueError(f'WebShop source file changed: {name}')
    # Sources are prepared at image build time; runtime must remain read-only.
    text_env = (source / 'web_agent_site/envs/web_agent_text_env.py').read_text()
    env_init = (source / 'web_agent_site/envs/__init__.py').read_text()
    if '\nimport torch\n' in text_env or 'from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv\n' in env_init.split('def __getattr__')[0]:
        raise ValueError('WebShop text-mode sources are not prepared; rebuild the official image')
    if str(source) not in sys.path: sys.path.insert(0,str(source))
    return source






class ProductStore(Mapping):
    """Disk-backed official product records; only visited products stay in memory."""
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        self.count = self.db.execute('SELECT COUNT(*) FROM products').fetchone()[0]
        self.lookup = lru_cache(maxsize=512)(self._lookup)

    def _lookup(self, asin):
        row = self.db.execute('SELECT payload FROM products WHERE asin=?', (asin,)).fetchone()
        if row is None: raise KeyError(asin)
        return json.loads(row[0])

    def __getitem__(self, asin):
        return self.lookup(asin)

    def __contains__(self, asin):
        return self.db.execute('SELECT 1 FROM products WHERE asin=?', (asin,)).fetchone() is not None

    def __iter__(self):
        for row in self.db.execute('SELECT asin FROM products ORDER BY ordinal'):
            yield row[0]

    def __len__(self):
        return self.count


class ProductSequence(Sequence):
    def __init__(self, store): self.store = store
    def __len__(self): return len(self.store)
    def __getitem__(self, index):
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0: index += len(self)
        if index < 0 or index >= len(self): raise IndexError(index)
        row = self.store.db.execute('SELECT asin FROM products WHERE ordinal=?', (index,)).fetchone()
        return self.store[row[0]]
    def __iter__(self):
        for asin in self.store: yield self.store[asin]


class ProductPrices(Mapping):
    def __init__(self, store): self.store = store
    def __len__(self): return len(self.store)
    def __iter__(self): return iter(self.store)
    def __getitem__(self, asin):
        row = self.store.db.execute('SELECT price FROM products WHERE asin=?', (asin,)).fetchone()
        if row is None: raise KeyError(asin)
        return row[0]


def engine_module(root):
    root = Path(root).resolve()
    prepare_source()
    sparse_search_class()
    utils = importlib.import_module('web_agent_site.utils')
    utils.DEFAULT_FILE_PATH = str(root/'raw/items_shuffle.json')
    utils.DEFAULT_ATTR_PATH = str(root/'raw/items_ins_v2.json')
    utils.HUMAN_ATTR_PATH = str(root/'raw/items_human_ins.json')
    engine = importlib.import_module('web_agent_site.engine.engine')
    engine.DEFAULT_ATTR_PATH = utils.DEFAULT_ATTR_PATH
    engine.HUMAN_ATTR_PATH = utils.HUMAN_ATTR_PATH
    return engine


@lru_cache(maxsize=2)
def runtime(root):
    root = raw_data_root(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if 'resource_directory' not in manifest:
        raise ValueError('Full WebShop assets are not prepared. Install the prepared WebShop resource bundle first.')
    folder = root/manifest['resource_directory']
    engine_module(root)
    module = importlib.import_module('web_agent_site.envs.web_agent_text_env')
    store = ProductStore(folder/'catalog.sqlite')
    goals = json.loads(store.db.execute("SELECT value FROM metadata WHERE key='goals'").fetchone()[0])
    # The upstream search, HTML transitions and reward function remain unchanged.
    # Replace only eager loading with equivalent prepared records and disk lookups.
    module.load_products = lambda **kwargs: (ProductSequence(store), store, ProductPrices(store), {})
    module.get_goals = lambda *args: copy.deepcopy(goals)
    module.init_search_engine = lambda **kwargs: sparse_search_class()(str(folder/'index'))
    state = random.getstate()
    try:
        random.seed(233)
        server = module.SimServer('http://127.0.0.1:3000', str(root/'raw/items_shuffle.json'),
                                  num_products=None, human_goals=True)
    finally:
        random.setstate(state)
    return module, server


def preflight(root):
    root = raw_data_root(root)
    manifest = json.loads((root/'manifest.json').read_text())
    module, server = runtime(str(root))
    if len(server.goals) != manifest['tasks']: raise ValueError('WebShop goal count changed; rebuild data')
    if len(server.all_products) != manifest['products']: raise ValueError('WebShop product count changed')
    if server.search_engine.num_docs != manifest['indexed_products']: raise ValueError('Incomplete WebShop search index')
    result = {'corpus_scope':manifest['corpus_scope'], 'products':len(server.all_products),
            'searchable_products':manifest['indexed_products'], 'empty_product_records':manifest['index_report']['empty'],
            'tasks':len(server.goals), 'tasks_by_split':manifest['tasks_by_split'],
            'split_rule':manifest['split'], 'human_demonstrations':manifest.get('human_demonstrations',{})}
    return result


class Adapter(TextAdapter):
    domain = DOMAIN
    system_prompt = SYSTEM_PROMPT
    tool_name = 'webshop_step'
    tool_description = ('Execute one WebShop action. For search, supply actual task-specific words, '
                        'for example search[blue cotton shirt], when search_available is true. '
                        'For clicks, copy a complete command from available_actions. '
                        'click[buy now] ends the task; verify requested specifications first.')
    tool_action_description = ('A complete action with actual values, e.g. search[blue cotton shirt] '
                               'or a click command copied from available_actions. Do not submit '
                               'search[keywords] or search[query]; those are unfilled templates.')

    def __init__(self, data_root, task_id, max_actions=50):
        root = raw_data_root(data_root)
        demonstration = demonstration_tasks(root).get(task_id)
        super().__init__(root, demonstration['source_task_ids'][0] if demonstration else task_id, max_actions)
        if demonstration:
            self.task = copy.deepcopy(demonstration)
            self.task_id, self.goal = task_id, self.task['goal']

    def start(self):
        module, server = runtime(str(self.root))
        demonstration = self.task.get('demonstration_goal')
        index = ('human' + self.task['task_id'].rsplit('_', 1)[-1]) if demonstration else self.task['goal_index']
        if not demonstration and server.goals[index]['instruction_text'] != self.goal:
            raise ValueError('WebShop task no longer matches the pinned engine goal')
        state = random.getstate()
        try:
            server.user_sessions.pop(str(index), None)
            with module.app.app_context():
                self.env = module.WebAgentTextEnv(observation_mode='text',server=server,get_image=False)
                # Constructor creates a random session; remove it before the chosen deterministic task.
                server.user_sessions.pop(self.env.session, None)
                if demonstration:
                    # Pin the original human task, including its original price
                    # ceiling. Never expose target ASIN/attributes to the agent.
                    server.user_sessions[str(index)] = {'goal': copy.deepcopy(demonstration), 'done': False}
                self.observation, _ = self.env.reset(session=index)
        finally:
            random.setstate(state)
        self.update_actions()

    def update_actions(self):
        available = self.env.get_available_actions()
        self.search_available = bool(available['has_search_bar']) and not self.done
        self.available_actions = [] if self.done else [
            f'click[{label}]' for label in available['clickables'] if label != 'search']

    def snapshot(self):
        state = super().snapshot()
        state['search_available'] = getattr(self, 'search_available', False) and not self.done
        if self.done:
            state['available_actions'] = []
        state['action_instructions'] = (
            'Write a search action using your own task-specific words, e.g. search[blue cotton shirt]. '
            'The search words are not limited to available_actions. '
            'available_actions lists complete click commands only.'
            if state['search_available'] else
            'Search is not available on this page. Choose a listed click command; '
            'return to search with click[back to search] when listed.'
        ) if not self.done else 'Task finished.'
        return state

    def perform(self, action):
        match = re.fullmatch(r'(search|click)\[(.*)\]', action.strip(), re.S)
        if not match or not match[2].strip(): raise ValueError('Provide a complete action, e.g. search[blue cotton shirt], or copy a listed click command')
        operation,value = match.groups()
        value = value.strip()
        if operation == 'search':
            if not self.search_available: raise ValueError('Return to a search page first')
            if value.lower() in {'keywords','keyword','query','search terms','search query','your query','your keywords'}:
                raise ValueError('You submitted a format placeholder. Replace it with actual words from the shopping task, for example search[blue cotton shirt].')
            if re.search(r'<[racq]>',value,re.I): raise ValueError('Internal debug search selectors are disabled')
        elif f'click[{value.lower()}]' not in self.available_actions:
            raise ValueError('Click a visible label from available_actions')
        module,_ = runtime(str(self.root))
        with module.app.app_context():
            self.observation,reward,done,_ = self.env.step(f'{operation}[{value}]')
        self.done = self.submitted = bool(done)
        self.score = float(reward) if done else 0.0
        self.success = bool(done and self.score == 1.0)
        self.update_actions()

    def close(self):
        if getattr(self,'env',None) is not None:
            self.env.server.user_sessions.pop(self.env.session,None)
            self.env = None




















