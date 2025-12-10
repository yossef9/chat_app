# Document Chat AI

A React application that lets users upload documents (PDF, DOCX, TXT) and chat with an AI that answers using only the uploaded documents.

This README focuses on how to fetch the `backend_and_frontend` branch and run the project locally on Windows (PowerShell / cmd.exe / Git Bash) and Unix/macOS.

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Fetch the branch](#fetch-the-branch)
- [Run the backend (FastAPI)](#run-the-backend-fastapi)
  - [Create venv (Unix/macOS)](#create-venv-unixmacos)
  - [Create venv (Windows PowerShell)](#create-venv-windows-powershell)
  - [Install dependencies](#install-dependencies)
  - [Create .env](#create-env)
  - [Run backend](#run-backend)
- [Run the frontend (React + Vite)](#run-the-frontend-react--vite)
- [Quick checklist](#quick-checklist)
- [Troubleshooting](#troubleshooting)
- [Useful commands summary](#useful-commands-summary)

---

## Prerequisites

- Git
- Node.js (v18+ recommended) and npm
- Python 3.10+ (3.11 preferred)
- Optional: WSL (Windows Subsystem for Linux) for easier Python dependency installs on Windows

## Fetch the branch

Open a terminal and run:

```bash
# go to the project directory (adjust path)
cd /path/to/document-chat-ai

# fetch and switch to the branch
git fetch origin
git checkout backend_and_frontend
git pull origin backend_and_frontend
```

## Run the backend (FastAPI)

Path: `backend/ai-docs-chat`

### Create venv (Unix/macOS)

```bash
cd backend/ai-docs-chat
python3 -m venv venv
source venv/bin/activate
```

### Create venv (Windows PowerShell)

```powershell
cd backend/ai-docs-chat
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Create venv (Windows cmd.exe)

```cmd
cd backend/ai-docs-chat
python -m venv venv
venv\Scripts\activate.bat
uvicorn src.main:app --reload  
```
## Run the frontend (React + Vite)

From the repository root:

```bash
npm install
npm run dev
```

Vite dev server is usually available at `http://localhost:5173`.

### Install dependencies
With activated venv
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Create `.env`

Create a `.env` file in `backend/ai-docs-chat` with the keys the backend needs. Example:

```env
SECRET_KEY = your jwt secret key
ALGORITHM = your password encription algorithm(ex. HS256)
ACCESS_TOKEN_EXPIRE_MINUTES = login expiration minutes
MONGO_URI=your_mongodb_connection_uri
```

Adjust keys and names based on your environment.





## Quick checklist

- Python venv created and activated in `backend/ai-docs-chat`
- `pip install -r requirements.txt` completed successfully
- `.env` created under `backend/ai-docs-chat` with correct keys
- Backend running: `uvicorn src.main:app --reload`
- From repo root: `npm install` and `npm run dev` completed

## Troubleshooting

- `uvicorn` not found — ensure venv is active and `uvicorn` is in `requirements.txt` or run `pip install uvicorn`.
- Windows binary dependency issues (e.g., `uvloop`) — use WSL or install compatible wheels.
- ESLint errors when running `npm run lint` — lint configuration or environment mismatch; Vite dev server may still run.
- Port conflicts — change the port for uvicorn or Vite (`--port` or Vite config).
- Frontend cannot reach backend — check CORS settings and ensure both servers run on expected ports.



