# Setup Guide for Complete Beginners

If you've never set up Python, used an API, or worked with environment variables, this guide is for you. It walks through every step in the order you need to do them, with screenshots and definitions of every term. Estimated time: **30 minutes**.

> **What you'll have at the end:** PathoMAN 2.0 running on your computer, classifying variants the same way it does in published clinical genomics tools — with no charges to your authors' accounts and no risk of leaking secrets.

---

## Part 1 — What is this? Glossary in plain English

| Term | What it means in this project |
|---|---|
| **Variant** | A specific change in a person's DNA, like `BRCA2:c.5946del`. PathoMAN reads one of these at a time and tells you whether it's likely cancer-causing. |
| **ACMG / AMP** | The American College of Medical Genetics and Genomics + the Association for Molecular Pathology. They wrote the rulebook (in 2015) for how to classify variants as Pathogenic, Likely Pathogenic, Variant of Uncertain Significance (VUS), Likely Benign, or Benign. |
| **LLM** | Large Language Model — like ChatGPT or Claude. PathoMAN uses one (Claude) to write plain-English explanations of its results, but the actual classification is done by Python rules, not the LLM. |
| **API key** | A long secret password that lets PathoMAN talk to Anthropic's Claude service over the internet. You get your own key for free, paste it into a file, and PathoMAN uses it. **Never share your key with anyone.** |
| **`.env` file** | A small text file called `.env` (with the dot at the start) that stores your API keys. It lives only on your computer and is **never** uploaded to GitHub. |
| **Python** | The programming language PathoMAN is written in. You'll install it once, then never think about it again. |
| **Terminal / Command line** | The black-screen text interface where you type commands like `python app.py`. On Mac it's called Terminal; on Windows it's called Command Prompt or PowerShell. |
| **Streamlit** | A library that turns Python code into a web app you can use in your browser. PathoMAN's user interface is built with it. |

---

## Part 2 — Get the software (one-time setup)

### Step 2.1 — Install Python (if you don't have it)

**On Mac:**
1. Open Terminal (Cmd+Space → type "Terminal" → Enter)
2. Type: `python3 --version` and press Enter
3. If you see something like `Python 3.10.x` or higher, skip to Step 2.2
4. If not, install it: download from https://www.python.org/downloads/ — pick the latest 3.11 or 3.12 version

**On Windows:**
1. Download Python from https://www.python.org/downloads/
2. **Important:** during install, check the box "Add Python to PATH"
3. Open Command Prompt (Win+R → type "cmd" → Enter)
4. Type: `python --version` and press Enter — you should see `Python 3.11.x` or similar

### Step 2.2 — Install Git (if you don't have it)

**On Mac:** type `git --version` in Terminal. If it asks to install, click yes.

**On Windows:** download Git from https://git-scm.com/download/win and use default settings.

### Step 2.3 — Download PathoMAN

In Terminal/Command Prompt:

```bash
cd Desktop                                         # or wherever you want it saved
git clone https://github.com/vigneshravi/PathoMAN2.0.git
cd PathoMAN2.0
```

This downloads the entire project to a folder called `PathoMAN2.0` on your Desktop.

### Step 2.4 — Set up a clean Python environment

We don't want PathoMAN's dependencies to mess with anything else on your computer. Use a "virtual environment" (called `.venv`):

**On Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

After activation, your terminal prompt should show `(.venv)` at the start. From now on, this means you're using the project's clean environment.

### Step 2.5 — Install the dependencies

```bash
pip install -r requirements.txt
```

This downloads about 600 MB of Python packages (LangChain, FAISS, etc.). Takes 3–8 minutes depending on your internet speed.

---

## Part 3 — Get your own API keys (free)

PathoMAN talks to two services that need keys: **Anthropic** (for the AI explanations) and **NCBI** (for ClinVar lookups). Both are free for the volume PathoMAN uses. You'll spend about $0.01 per variant on Anthropic; NCBI is free.

### Step 3.1 — Get an Anthropic API key

1. Go to https://console.anthropic.com/
2. Sign up with your email (or log in if you have an account)
3. Click your profile (top right) → **API Keys**
4. Click **Create Key**, give it a name like `pathoman-personal`
5. **Copy the key right away** — Anthropic only shows it once. It looks like `sk-ant-api03-AbCdEf123...` (very long).
6. **Add credit:** Settings → **Plans & Billing** → add at least $5. Classifying 10 variants costs about $0.10.

> ⚠ **Never share this key.** If someone else uses it, they'll spend YOUR credits. If you accidentally paste it somewhere public (like GitHub or a screenshot), revoke it immediately at the same Anthropic page and create a new one.

### Step 3.2 — Get a free NCBI API key

NCBI = National Center for Biotechnology Information (US government). The key is free and just makes their database faster for you.

1. Go to https://www.ncbi.nlm.nih.gov/account/
2. Create a free account (or sign in)
3. Click your username (top right) → **Account Settings**
4. Scroll to **API Key Management** → **Create an API Key**
5. Copy the 32-character key

You'll also need an email address NCBI can contact if there's a problem with your queries. Use the same email you registered with.

### Step 3.3 — Put your keys in a `.env` file

In the `PathoMAN2.0` folder, you'll see a file called `.env.example`. Make a copy of it called `.env` (literally just `.env` — no other extension):

**On Mac/Linux:**
```bash
cp .env.example .env
```

**On Windows:**
```bash
copy .env.example .env
```

Now open `.env` in any text editor (TextEdit on Mac, Notepad on Windows, or VS Code). It looks like:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
NCBI_API_KEY=your_ncbi_api_key_here
NCBI_EMAIL=your_email@example.com
```

Replace each `your_..._here` with your actual key/email. The lines should look like:

```
ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf123...   (your real key)
NCBI_API_KEY=a8d568868577f61ff78334aaa35a3d39fa08   (your real key, 32 characters)
NCBI_EMAIL=jane.doe@university.edu             (your real email)
```

**Save the file.** Close it.

> ✅ **Why is this safe?** The file `.env` is listed in `.gitignore`, which means git will never upload it to GitHub. Your keys stay on your computer only.

---

## Part 4 — Run PathoMAN

In your activated terminal (with `(.venv)` showing at the start of the prompt), from the `PathoMAN2.0` folder:

```bash
streamlit run app.py
```

After 5–10 seconds, your default web browser should open to `http://localhost:8501` showing PathoMAN's pac-man interface.

If the browser doesn't open automatically, paste `http://localhost:8501` into a browser yourself.

### Try a variant

1. In the input box, type: `BRCA2:c.5946delT`
2. Click the **Look Up** button → confirm transcript selection
3. Click **Classify**
4. Wait ~30 seconds for the pipeline to run all 10 nodes
5. You'll see the classification (Likely Pathogenic), the criteria used, the SVI overrides applied, and a Plain English explanation tab

---

## Part 5 — Common errors and fixes

| Error in terminal | Likely cause | Fix |
|---|---|---|
| `streamlit: command not found` | `.venv` not activated | Run `source .venv/bin/activate` (Mac) or `.venv\Scripts\activate` (Windows) |
| `ModuleNotFoundError: No module named 'langchain'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `ANTHROPIC_API_KEY: <missing>` in app | `.env` file not present or empty | Verify `.env` exists in the project root with your key inside |
| `API key invalid` from NCBI | Empty or wrong NCBI key | Double-check `NCBI_API_KEY=` line in `.env` has your 32-char key |
| Browser shows blank page | Streamlit still starting | Wait another 10 seconds, refresh |
| `429 Too Many Requests` on Anthropic | Hit rate limit | Wait a minute and try again; ensure your account has credit |
| Knowledge base empty | First-run RAG index not built | Run `python -m svi.bootstrap_kb` once to fetch the 6 SVI/VCEP papers |

If something else goes wrong, copy the full error from your terminal and search for it on the project's GitHub Issues page: https://github.com/vigneshravi/PathoMAN2.0/issues

---

## Part 6 — Stopping PathoMAN

Press `Ctrl+C` in the terminal where Streamlit is running. To leave the virtual environment, type `deactivate`.

To run PathoMAN again later:
```bash
cd ~/Desktop/PathoMAN2.0      # or wherever you saved it
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
streamlit run app.py
```

---

## Part 7 — How to cite this work in your paper / poster / talk

If you use PathoMAN in any academic, research, or clinical-presentation context, you must cite it. The repository includes a `CITATION.cff` file that GitHub automatically converts to BibTeX, RIS, APA, and other formats — click the **"Cite this repository"** button on the GitHub page.

**Minimum APA citation:**

> Ravichandran, V. (2026). *PathoMAN 2.0: Pathogenicity of Mutation Analyzer — A hybrid retrieval-augmented agentic pipeline for ACMG/AMP germline variant classification*. Department of Health Informatics, Rutgers School of Health Professions. https://github.com/vigneshravi/PathoMAN2.0

If you use it in software (a fork, a derivative project, a clinical pipeline), the LICENSE file requires you to:
1. Keep the NOTICE file at the project root unchanged
2. Cite this repo from your README
3. For commercial use, contact the author for written permission

---

## Part 8 — One-screen summary

```
1. Install Python 3.11+ and Git
2. git clone https://github.com/vigneshravi/PathoMAN2.0.git
3. cd PathoMAN2.0
4. python3 -m venv .venv && source .venv/bin/activate
5. pip install -r requirements.txt
6. Get free Anthropic key (https://console.anthropic.com/)
7. Get free NCBI key (https://www.ncbi.nlm.nih.gov/account/)
8. cp .env.example .env  → paste your keys into .env
9. python -m svi.bootstrap_kb     # builds knowledge base, one time
10. streamlit run app.py
11. Browser opens → type a variant → classify
```

---

*This guide is part of PathoMAN 2.0. Cite this repository if you use it. License: Apache 2.0 with attribution requirement (see LICENSE).*
