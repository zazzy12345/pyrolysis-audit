"""Download the pinned public inputs and verify every file against SHA-256."""
from pathlib import Path
import base64, hashlib, json, urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    m=json.loads((ROOT/'public_data_manifest.json').read_text(encoding='utf-8'))
    for row in m['files']:
        dst=ROOT/row['destination']
        if dst.exists():
            data=dst.read_bytes()
        else:
            path=row['repository_path']
            url=f"https://raw.githubusercontent.com/{m['repository']}/{m['commit']}/{path}"
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'pyrolysis-audit-v0.1'})
                with urllib.request.urlopen(req,timeout=45) as response:data=response.read()
            except Exception:
                url=f"https://api.github.com/repos/{m['repository']}/contents/{path}?ref={m['commit']}"
                req=urllib.request.Request(url,headers={'User-Agent':'pyrolysis-audit-v0.1'})
                with urllib.request.urlopen(req,timeout=45) as response:data=base64.b64decode(json.load(response)['content'])
        actual=hashlib.sha256(data).hexdigest()
        if actual!=row['sha256']:
            raise ValueError(f"Input hash mismatch: {row['destination']}; expected {row['sha256']}, got {actual}")
        if len(data)!=row['bytes']:raise ValueError('Input size mismatch: '+row['destination'])
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(data)
        print('verified '+row['destination'],flush=True)
    print(f"Verified {len(m['files'])} inputs from {m['repository']} at {m['commit']}")

if __name__=='__main__':main()
