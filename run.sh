#!/bin/bash
cd "$(dirname "$0")/backend"
rm -f linkage.db
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
