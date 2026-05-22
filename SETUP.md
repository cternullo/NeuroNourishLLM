# NeuroNourishLLM — Team Setup Guide

## What This Is
A shared Groq-powered knowledge base that integrates with Obsidian.
Ingest research papers, Wikipedia topics, and URLs — then chat with 
the knowledge base using natural language.

## Prerequisites

### Everyone needs:
- Python 3.10+ — python.org/downloads
- Git — git-scm.com/downloads
- Obsidian — obsidian.md
- A free Groq API key — console.groq.com

### Verify your installs:
```bash
python3 --version
git --version
```

## Step 1 — Clone the Repo
```bash
git clone https://github.com/cternullo/NeuroNourishLLM.git
cd NeuroNourishLLM
```

## Step 2 — Get Your Groq API Key
1. Go to console.groq.com
2. Sign up (free, no credit card)
3. Go to API Keys → Create new key
4. Copy it — you will need it in the next step

## Step 3 — Create Your .env File
Create a file called .env in the project root. Never commit this file.

For your vault path:
- Mac: /Users/yourname/NeuroNourishLLM
- Windows: C:\Users\yourname\NeuroNourishLLM

## Step 4 — Open as Obsidian Vault
1. Open Obsidian
2. Click Open folder as vault
3. Navigate to your cloned NeuroNourishLLM folder
4. Click Open
5. Click Trust and enable all plugins if prompted

## Step 5 — Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 6 — Test Your Setup
```bash
python cli.py ingest topic "Apolipoprotein B"
```
If it works, a new markdown note will appear in your Obsidian vault 
under the wiki/ folder within seconds.

---

## Daily Commands

### Ingest a Wikipedia topic
```bash
python cli.py ingest topic "your topic here"
```

### Ingest a URL
```bash
python cli.py ingest url https://example.com/article
```

### Ingest a PDF research paper
```bash
python cli.py ingest pdf /path/to/paper.pdf
```

### Chat with the knowledge base
```bash
python cli.py chat "your question here"
```

### Regenerate the index
```bash
python cli.py index
```

---

## Team Git Workflow

### Before you start working — always pull first:
```bash
git pull
```

### After adding notes or papers — push your changes:
```bash
git add .
git commit -m "added paper on LDL and cognitive decline"
git push
```

### Rules:
- Never commit your .env file
- Always pull before you push
- Write a clear commit message describing what you added

---

## Troubleshooting

### command not found: python3
Install Python from python.org/downloads — use the installer for 
your OS.

### ModuleNotFoundError
Run pip install -r requirements.txt again.

### GROQ_API_KEY not found
Check your .env file exists in the project root and contains your key.

### Obsidian vault is empty
Make sure VAULT_PATH in your .env points exactly to the cloned folder.
Run python cli.py index to regenerate the note index.
