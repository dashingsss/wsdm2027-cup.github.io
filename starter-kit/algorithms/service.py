"""Algorithm-only HTTP service. Run inside the participant container."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from algorithms.protocol import (VERSION, MAX_MESSAGE_BYTES, state_from_json,
                                 guidance_to_json, guidance_from_json)


def algorithm_worker(connection, settings):
    if hasattr(os, 'setsid'):
        os.setsid()
    try:
        from algorithms import create_algorithm
        method = Path(settings['method'])
        root = Path(__file__).resolve().parent
        method = (root.parent / method).resolve()
        if not method.is_relative_to(root):
            raise ValueError('Method must be inside algorithms/')
        algorithm = create_algorithm(method, options=settings['options'],
                                     seed=settings['seed'], domain=settings['domain'])
        connection.send_bytes(b'{"ok":true,"value":null}')
        corpus = None
        while True:
            message = json.loads(connection.recv_bytes(MAX_MESSAGE_BYTES))
            operation, payload = message['operation'], message['payload']
            if operation == 'prepare':
                corpus = payload['corpus'] if algorithm.requires_corpus else []
                algorithm.prepare(corpus)
                value = None
            elif operation == 'reset':
                if corpus is None:
                    raise ValueError('Prepare the corpus before starting a task')
                algorithm.reset(state_from_json(payload['state']), corpus)
                value = None
            elif operation == 'guidance':
                value = guidance_to_json(algorithm.get_guidance(state_from_json(payload['state'])))
                guidance_from_json(value)
            else:
                raise ValueError('Unknown operation')
            raw = json.dumps({'ok': True, 'value': value}, allow_nan=False).encode()
            if len(raw) > MAX_MESSAGE_BYTES:
                raise ValueError('Response exceeds transport safety bound')
            connection.send_bytes(raw)
    except BaseException as exc:
        try:
            connection.send_bytes(json.dumps({'ok': False, 'error': type(exc).__name__}).encode())
        except (OSError, BrokenPipeError):
            pass
    finally:
        connection.close()


class Runtime:
    def __init__(self):
        self.process = self.connection = None
        self.gate = threading.Lock()

    def stop(self):
        process = self.process
        if process is not None:
            if process.is_alive():
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
            process.join(timeout=2)
        self.process = None
        # A concurrent request may still be unwinding its connection read.

    def call(self, operation, payload):
        if operation == 'health':
            return {'ok': True, 'value': {'version': VERSION}}
        if operation == 'cancel':
            self.stop()
            return {'ok': True, 'value': None}
        if not self.gate.acquire(blocking=False):
            return {'ok': False, 'error': 'Busy'}
        try:
            if operation == 'initialize':
                if set(payload) != {'version', 'method', 'domain', 'seed', 'options'} or payload['version'] != VERSION:
                    raise ValueError('Incompatible initialization')
                self.stop()
                if self.connection:
                    self.connection.close()
                ctx = multiprocessing.get_context('spawn')
                self.connection, child = ctx.Pipe()
                self.process = ctx.Process(target=algorithm_worker, args=(child, payload))
                self.process.start()
                child.close()
            elif operation in ('prepare', 'reset', 'guidance'):
                if self.process is None or not self.process.is_alive():
                    raise ValueError('Algorithm worker is unavailable')
                self.connection.send_bytes(json.dumps({'operation': operation, 'payload': payload},
                                                     allow_nan=False).encode())
            else:
                raise ValueError('Unknown operation')
            return json.loads(self.connection.recv_bytes(MAX_MESSAGE_BYTES))
        finally:
            self.gate.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '-1'))
            if not 0 <= length <= MAX_MESSAGE_BYTES:
                raise ValueError('Invalid message size')
            self.connection.settimeout(30)
            payload = json.loads(self.rfile.read(length))
            result = self.server.runtime.call(self.path.removeprefix('/'), payload)
        except Exception as exc:
            result = {'ok': False, 'error': type(exc).__name__}
        raw = json.dumps(result, allow_nan=False).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.runtime = Runtime()
    try:
        server.serve_forever()
    finally:
        server.runtime.stop()
        server.server_close()


if __name__ == '__main__':
    main()
