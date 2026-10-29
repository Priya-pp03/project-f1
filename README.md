# F1 circuits: chunk + embed

Turns `circuits.csv` (77 Formula 1 circuits) into retrieval-ready chunks and
embeddings using the OpenAI embeddings API.

## Pipeline

```
circuits.csv  ->  chunk_circuits.py  ->  data/chunks.jsonl
              ->  embed_chunks.py     ->  data/embeddings.npy
                                          data/embeddings_meta.json
```

Each CSV row becomes **one chunk**: a short natural-language document
(name, location, coordinates, altitude, reference, Wikipedia link) plus a
metadata block. Row order in `embeddings.npy` matches line order in
`chunks.jsonl`, and `embeddings_meta.json` records the mapping.

## Setup

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # then put your real key in .env
```

`embed_chunks.py` and `search.py` read `OPENAI_API_KEY` from the environment or
from `.env`.

## Choosing an endpoint

The endpoint is inferred from the key prefix, so a **Pinecone** key (`pcsk_`) is
rejected immediately with an explanation, an **OpenAI** key (`sk-proj-…`) uses
`api.openai.com`, and an **OpenRouter** key (`sk-or-v1-…`) uses
`https://openrouter.ai/api/v1`.

Override either when using another OpenAI-compatible gateway:

```powershell
.\venv\Scripts\python.exe embed_chunks.py `
  --base-url https://my-gateway/v1 `
  --model my-embedding-model
```

Environment equivalents: `OPENAI_BASE_URL` and `EMBED_MODEL`.

Note that `search.py` reuses the endpoint recorded in the manifest, so query
vectors are always produced by the same model that built the index.

## Run

```powershell
.\venv\Scripts\python.exe chunk_circuits.py
.\venv\Scripts\python.exe embed_chunks.py
.\venv\Scripts\python.exe search.py "street circuit in the united states"
```

Useful flags:

- `chunk_circuits.py --input circuits.csv --output data/chunks.jsonl`
- `embed_chunks.py --model text-embedding-3-large --dimensions 1024 --batch-size 100`

## Output artifacts

| File | Contents |
| --- | --- |
| `data/chunks.jsonl` | One JSON object per chunk: `id`, `text`, `metadata` |
| `data/embeddings.npy` | `float32` array of shape `(n_chunks, dim)` |
| `data/embeddings_meta.json` | Model, dimension, count, and chunk id/metadata per row |

`data/` is git-ignored because it is regenerable.
