"""Standalone BGE trajectory retrieval: retrieve once per task."""
from pathlib import Path
from algorithms.base import Guidance, PlanningAlgorithm

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
MODEL_REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
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


class BGEEncoder:
    """Official unquantized BGE ONNX weights, CLS pooling and L2 normalization.

    Matches the model card's retrieval recipe: instruction on queries only.
    CPU threads default to one to avoid monopolizing the shared Mac.
    """
    def __init__(self, model_path=None, batch_size=16, threads=1):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError('Dense requires numpy, onnxruntime, tokenizers and huggingface_hub from requirements.txt') from error
        self.np = np
        self.batch_size = positive_int(batch_size, 'batch_size')
        positive_int(threads, 'threads')
        if model_path:
            root = Path(model_path).expanduser().resolve()
        else:
            root = Path(snapshot_download(MODEL_NAME, revision=MODEL_REVISION,
                        allow_patterns=['onnx/model.onnx', 'tokenizer.json']))
        if not (root/'onnx/model.onnx').is_file() or not (root/'tokenizer.json').is_file():
            raise ValueError('model_path must contain onnx/model.onnx and tokenizer.json')
        self.tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.max_tokens = 512
        self.pad_id = self.tokenizer.token_to_id('[PAD]')
        if self.pad_id is None:
            raise ValueError('Expected the BGE tokenizer with a [PAD] token')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root/'onnx/model.onnx'), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        if not self.inputs <= {'input_ids', 'attention_mask', 'token_type_ids'}:
            raise ValueError('Unexpected BGE ONNX input schema')

    def clip(self, text, budget):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return self.tokenizer.decode(ids[:max(0, budget)], skip_special_tokens=False) if len(ids) > budget else text

    def state_text(self, goal, observation, query=False):
        prefix = f'Task:\n{self.clip(goal, 192)}\n\nObservation:\n'
        # Reserve query instruction space on both sides for equal truncation.
        overhead = len(self.tokenizer.encode(QUERY_PREFIX + prefix).ids)
        obs = self.clip(observation, max(0, self.max_tokens - overhead - 8))
        while len(self.tokenizer.encode(QUERY_PREFIX + prefix + obs).ids) > self.max_tokens:
            obs = self.clip(obs, max(0, len(self.tokenizer.encode(obs, add_special_tokens=False).ids) - 8))
            if not obs and overhead > self.max_tokens:
                raise ValueError('State headers exceed encoder budget')
        return (QUERY_PREFIX if query else '') + prefix + obs

    def query(self, state):
        return self.state_text(state.goal, state.observation, query=True)

    def encode(self, texts):
        np = self.np
        vectors = []
        # A private tokenizer copy allows batch padding/truncation without
        # changing the tokenizer used to measure query budgets.
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_str(self.tokenizer.to_str())
        tokenizer.enable_truncation(max_length=self.max_tokens)
        tokenizer.enable_padding(pad_id=self.pad_id, pad_token='[PAD]')
        for start in range(0, len(texts), self.batch_size):
            encoded = tokenizer.encode_batch(texts[start:start+self.batch_size])
            arrays = {'input_ids': np.asarray([e.ids for e in encoded], dtype=np.int64),
                      'attention_mask': np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
                      'token_type_ids': np.asarray([e.type_ids for e in encoded], dtype=np.int64)}
            output = self.session.run(None, {k: arrays[k] for k in self.inputs})[0]
            if output.ndim != 3 or output.shape[-1] != 384:
                raise ValueError('Expected BGE-small token embeddings with dimension 384')
            vector = output[:, 0, :].astype(np.float32)  # CLS, not mean pooling
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            if not np.isfinite(vector).all() or (norms <= 0).any():
                raise ValueError('Invalid encoder output')
            vectors.append(vector / norms)
        return np.concatenate(vectors, axis=0)


class DenseStatic(PlanningAlgorithm):
    def __init__(self, top_k=1, max_guidance_chars=8000,
                 model_path=None, batch_size=16, threads=1):
        self.top_k = positive_int(top_k, 'top_k')
        self.max_guidance_chars = positive_int(max_guidance_chars, 'max_guidance_chars')
        self.encoder_options = dict(model_path=model_path, batch_size=positive_int(batch_size, 'batch_size'),
                                    threads=positive_int(threads, 'threads'))
        self.encoder = self.corpus = self.vectors = self.task_id = None
        self.guidance = Guidance()

    def prepare(self, corpus):
        validate_corpus(corpus)
        if self.encoder is None:
            self.encoder = BGEEncoder(**self.encoder_options)
        texts = [self.encoder.state_text(row['goal'], row['observations'][0]) for row in corpus]
        self.vectors = self.encoder.encode(texts)
        self.corpus, self.task_id = corpus, None

    def retrieve(self, state):
        eligible = candidates(self.corpus, state)
        scores = self.vectors @ self.encoder.encode([self.encoder.query(state)])[0]
        selected = sorted(eligible, key=lambda i: (-float(scores[i]), self.corpus[i]['task_id']))[:self.top_k]
        return format_guidance([self.corpus[i] for i in selected], self.max_guidance_chars)

    def reset(self, initial, corpus):
        self.task_id = None
        self.guidance = Guidance()
        if self.corpus is not corpus:
            self.prepare(corpus)
        candidates(corpus, initial)
        self.task_id = initial.task_id
        self.guidance = self.retrieve(initial)

    def get_guidance(self, state):
        if self.task_id != state.task_id:
            raise ValueError('Call reset before requesting guidance for a new task')
        return self.guidance


def create_algorithm(config):
    opts = config.get('options', {})
    return DenseStatic(top_k=config.get('top_k', 1),
                       max_guidance_chars=opts.get('max_guidance_chars', 8000),
                       model_path=opts.get('model_path'), batch_size=opts.get('batch_size', 16),
                       threads=opts.get('threads', 1))
