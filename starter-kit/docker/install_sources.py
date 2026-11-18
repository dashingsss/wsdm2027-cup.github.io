"""Fetch pinned upstream sources and prepare the read-only official image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def checkout(url, commit, directory):
    def git(*args):
        try:
            return subprocess.run(['git', '-C', str(directory), *args], check=True,
                                  capture_output=True, text=True, timeout=240).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Upstream Git operation failed: {exc.stderr.strip()}') from exc
    directory.mkdir()
    git('init', '--quiet')
    git('remote', 'add', 'origin', url)
    git('fetch', '--quiet', '--depth=1', 'origin', commit)
    git('checkout', '--quiet', '--detach', 'FETCH_HEAD')
    if git('rev-parse', 'HEAD') != commit:
        raise ValueError('Unexpected upstream commit')


def prepare_webshop(source):
    # The same two import-only changes used in the validated text environment.
    path = source / 'web_agent_site/envs/web_agent_text_env.py'
    text = path.read_text()
    if '\nimport torch\n' not in text:
        raise ValueError('Unexpected upstream Torch import; review the pinned source')
    text = text.replace('\nimport torch\n', '\n')
    text = text.replace("        if self.kwargs.get('get_image', 0):\n",
                        "        if self.kwargs.get('get_image', 0):\n            import torch  # Optional image features; text mode needs no torch\n")
    text = text.replace('        return torch.zeros(512)', '        import torch\n        return torch.zeros(512)')
    path.write_text(text)
    path = source / 'web_agent_site/envs/__init__.py'
    text = path.read_text()
    eager = 'from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv\n'
    if eager not in text:
        raise ValueError('Unexpected upstream environment imports')
    text = text.replace(eager, '')
    text = text.rstrip() + '\n\ndef __getattr__(name):\n    if name == "WebAgentSiteEnv":\n        from .web_agent_site_env import WebAgentSiteEnv\n        return WebAgentSiteEnv\n    raise AttributeError(name)\n'
    path.write_text(text)


def export_verified(source, destination, spec):
    destination.mkdir()
    for name, expected in spec['files'].items():
        member = (source / name).resolve()
        if not member.is_relative_to(source.resolve()) or not member.is_file():
            raise ValueError('Invalid upstream source path')
        if hashlib.sha256(member.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Upstream source checksum mismatch: {name}')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(member, target)
    # Keep upstream attribution and distribution terms when present.
    for member in source.iterdir():
        if member.is_file() and member.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE', 'README')):
            shutil.copy2(member, destination / member.name)
    manifest = {'commit': spec['commit'], 'url': spec['url'], 'files': spec['files']}
    (destination / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


def install(destination, lock):
    destination.mkdir(parents=True, exist_ok=True)
    for name in lock:
        if (destination / name).exists():
            raise ValueError(f'Refusing to overwrite existing sources: {destination / name}')
    # Stage both packages first; a failed download never installs a partial set.
    with tempfile.TemporaryDirectory(prefix='.source-build-', dir=destination) as temporary:
        work = Path(temporary)
        prepared = work / 'prepared'
        prepared.mkdir()
        for name, spec in lock.items():
            print(f'Fetching {name} at {spec["commit"]}', flush=True)
            source = work / name
            checkout(spec['url'], spec['commit'], source)
            if name == 'webshop':
                prepare_webshop(source)
            export_verified(source, prepared / name, spec)
        for name in lock:
            shutil.move(str(prepared / name), str(destination / name))
    print('Pinned source hashes verified; text-mode sources are ready.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name('source-lock.json').read_text())
    install(args.destination.resolve(), lock)
