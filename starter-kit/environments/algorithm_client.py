"""Trusted JSON client. Never imports or deserializes participant Python code."""
import json
import urllib.request
import urllib.error
from algorithms.base import PlanningAlgorithm
from algorithms.protocol import VERSION, MAX_MESSAGE_BYTES, state_to_json, guidance_from_json


class AlgorithmError(RuntimeError):
    pass


class AlgorithmUnavailable(AlgorithmError):
    """Transport failure of unknown attribution; do not silently score it as a wrong answer."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AlgorithmError('Algorithm redirects are not allowed')


class RemoteAlgorithm(PlanningAlgorithm):
    requires_corpus = True

    def __init__(self, endpoint, method, *, domain, seed=42, options=None):
        self.endpoint = endpoint.rstrip('/')
        if not self.endpoint.startswith('http://'):
            raise ValueError('Use the private HTTP algorithm endpoint')
        self.setup = {'version': VERSION, 'method': str(method), 'domain': domain,
                      'seed': seed, 'options': options or {}}
        # Do not inherit a host proxy that could receive task observations.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, operation, payload=None, timeout=None):
        data = json.dumps(payload or {}, allow_nan=False).encode()
        if len(data) > MAX_MESSAGE_BYTES:
            raise AlgorithmError('Algorithm request exceeds the transport safety bound')
        req = urllib.request.Request(self.endpoint + '/' + operation, data=data,
                                     headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(req, timeout=timeout) as response:
                raw = response.read(MAX_MESSAGE_BYTES + 1)
            if len(raw) > MAX_MESSAGE_BYTES:
                raise AlgorithmError('Algorithm response exceeds the transport safety bound')
            result = json.loads(raw)
            if not isinstance(result, dict) or result.get('ok') is not True:
                raise AlgorithmError('Algorithm returned an error')
            return result.get('value')
        except (OSError, urllib.error.URLError) as exc:
            # Do not reflect arbitrary participant error text into official prompts.
            raise AlgorithmUnavailable('Algorithm communication failed: ' + type(exc).__name__) from None
        except ValueError:
            raise AlgorithmError('Algorithm response is not valid JSON') from None

    def prepare(self, corpus):
        self.request('initialize', self.setup)  # No index-preparation deadline.
        self.request('prepare', {'corpus': corpus})

    def reset(self, initial, corpus):
        self.request('reset', {'state': state_to_json(initial)})

    def get_guidance(self, state):
        try:
            return guidance_from_json(self.request('guidance', {'state': state_to_json(state)}))
        except ValueError as exc:
            raise AlgorithmError(str(exc)) from None

    def cancel(self):
        return self.request('cancel', timeout=5)
