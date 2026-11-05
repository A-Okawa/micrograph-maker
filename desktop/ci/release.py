# -*- coding: utf-8 -*-
"""GitHub releases for the PC app, through the REST API (GITHUB_TOKEN of the workflow).

  python release.py ensure TAG TITLE DRAFT(true|false) NOTES_FILE [draft-only]
        → prints the release id (creates it when missing; fills an empty description from NOTES_FILE).
          draft-only: prints nothing when the release exists and is already published (it is not changed then)
  python release.py upload RELEASE_ID FILE...                        (replaces assets of the same name)
  python release.py publish RELEASE_ID                               (draft → published, marked as the latest)
"""
import json
import os
import sys
import urllib.parse
import urllib.request

API = os.environ.get('GITHUB_API_URL', 'https://api.github.com')
REPO = os.environ['GITHUB_REPOSITORY']
TOKEN = os.environ['GH_TOKEN']


def call(method, url, data=None, ctype='application/json'):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    req = urllib.request.Request(url, data=body, method=method, headers={
        'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github+json', 'Content-Type': ctype,
        'X-GitHub-Api-Version': '2022-11-28'})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def releases():
    out, page = [], 1
    while True:
        got = call('GET', f'{API}/repos/{REPO}/releases?per_page=100&page={page}')
        out += got
        if len(got) < 100:
            return out
        page += 1


def ensure(tag, title, draft, notes_file, only_draft=False):
    notes = open(notes_file, encoding='utf-8').read() if notes_file and os.path.exists(notes_file) else ''
    sha = os.environ.get('GITHUB_SHA', '')                 # a draft's tag is made at this commit when it is published
    for r in releases():                                   # a draft renamed to untagged-… is found by its title
        if r['tag_name'] == tag or (r['draft'] and r['tag_name'].startswith('untagged-') and r['name'] == title):
            if only_draft and not r['draft']:
                return ''
            change = {}
            if notes and not (r.get('body') or '').strip():
                change['body'] = notes
            if r['draft'] and sha:
                change['target_commitish'] = sha
            if change or r['tag_name'] != tag:            # always with tag_name: without it GitHub renames a draft's tag
                change['tag_name'] = tag                   # to untagged-…
                call('PATCH', f'{API}/repos/{REPO}/releases/{r["id"]}', change)
            return r['id']
    new = {'tag_name': tag, 'name': title, 'body': notes, 'draft': draft == 'true', 'prerelease': False}
    if sha:
        new['target_commitish'] = sha
    return call('POST', f'{API}/repos/{REPO}/releases', new)['id']


def upload(rid, files):
    have = {a['name']: a['id'] for a in call('GET', f'{API}/repos/{REPO}/releases/{rid}/assets?per_page=100')}
    for f in files:
        name = os.path.basename(f)
        if name in have:
            call('DELETE', f'{API}/repos/{REPO}/releases/assets/{have[name]}')
        with open(f, 'rb') as fh:
            data = fh.read()
        url = f'https://uploads.github.com/repos/{REPO}/releases/{rid}/assets?name={urllib.parse.quote(name)}'
        call('POST', url, data, 'application/octet-stream')
        print('uploaded', name, len(data), 'bytes')


def publish(rid):
    call('PATCH', f'{API}/repos/{REPO}/releases/{rid}', {'draft': False, 'make_latest': 'true'})
    print('published', rid)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'ensure':
        print(ensure(*sys.argv[2:6], only_draft=sys.argv[6:7] == ['draft-only']))
    elif cmd == 'upload':
        upload(sys.argv[2], sys.argv[3:])
    elif cmd == 'publish':
        publish(sys.argv[2])
