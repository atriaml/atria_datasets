# Dataset Manifest

A small local explorer for prepared datasets discovered from cache snapshots. The UI has three routes:

- `/` — inventory of prepared datasets under the configured output root
- `/prepare` — background preparation jobs for generating new cached datasets
- `/datasets/:datasetId/documents/:split/:index` — dedicated document page with
  image, metadata, transcription, validation warnings, overlays, and OCR hierarchy

The frontend uses React, TypeScript, Vite, Tailwind CSS, React Router, and
local shadcn-style UI primitives. The backend uses FastAPI, snapshot discovery
from `atria_core`, and the public lazy dataset factories for preparation jobs.

## Setup

From the repository root:

```bash
bash explorer/setup.sh
```

## Run

```bash
bash explorer/run.sh
```

Open <http://127.0.0.1:5173>. Both servers bind to localhost. The Vite server
proxies `/api` requests to FastAPI at `127.0.0.1:8000`.

## Using the explorer

1. On the inventory page, choose the output root to scan for prepared datasets.
   Leave it empty to use the default Atria datasets cache.
2. Click any discovered dataset card to open its documents and then a dedicated
   document page.
3. Use the prepare page when you want to build a new dataset in the background.
4. Add configuration overrides as JSON when needed, for example:

   ```json
   {"collection": "u17"}
   ```

The explorer does not edit annotations. It visualizes boxes or polygons and
reports basic problems such as missing text or dangling hierarchy parents.

## Development

Run the services separately when needed:

```bash
.venv/bin/uvicorn explorer.backend.app:app --reload --host 127.0.0.1 --port 8000
npm --prefix explorer/frontend run dev
```

Build the frontend:

```bash
npm --prefix explorer/frontend run build
```
