# Purge-Simulator_2

This repo contains your **legacy wizard UI pages** plus a newer simulation engine module.
If you feel stuck with GitHub/Codex, this file is your literal script.

## 🚨 If you only do one thing, do this

Run this command first:

```bash
python start_here.py
```

That script checks your setup and prints the exact next command.

---

## 0) Super-simple orientation (what these files are)

- `start_here.py` → your **coach script**. Run this first every time.
- `wizard_launcher.py` → opens the current legacy wizard pages in one app.
- `page2_pipe.py` → pipe/fluid inputs page.
- `page3_profile.py` → profile loading and validation page.
- `page4_purgesetup.py` → purge window + endpoint behavior page.
- `page5_simulationsetup.py` → simulation + nitrogen setup pages.
- `page6_summary.py` → simulation engine logic.

---

## 1) Total step-by-step (copy/paste version)

### Step 1 — open terminal in this repo

You should be in this folder:

```bash
/workspace/Purge-Simulator_2
```

Check:

```bash
pwd
```

### Step 2 — run the helper

```bash
python start_here.py
```

If it says modules are missing, run the printed `pip install ...` command.

### Step 3 — launch the wizard app

```bash
python wizard_launcher.py
```

### Step 4 — click through pages in this exact order

1. **Pipe & Fluid Inputs**
2. **Profile Loader**
3. **Purge Setup**
4. **Simulation Setup**
5. **Nitrogen Setup**

### Step 5 — write down what happens

Use a tiny template (copy this into a note):

- Page:
- What I clicked:
- What I expected:
- What happened:
- Error message (if any):

### Step 6 — commit your work (when you edit files)

```bash
git add .
git commit -m "describe what you changed"
git push
```

---

## 2) First-time GitHub workflow in plain English

1. Edit files.
2. Run a quick syntax check:

   ```bash
   python -m py_compile wizard_launcher.py start_here.py
   ```

3. Save your work with Git:

   ```bash
   git add .
   git commit -m "short message"
   git push
   ```

4. Open a Pull Request in GitHub.

---

## 3) Rebuild plan (after you can open all pages)

### Phase 1 — stabilize each page
- Make sure every page opens cleanly.
- Give each page a `get_data()` that returns validated inputs.

### Phase 2 — connect pages with shared state
- One shared dictionary for all page inputs.
- Next/Back saves/restores values.

### Phase 3 — run simulation
- Add a results page that calls `run_simulation(...)` from `page6_summary.py`.

### Phase 4 — export and confidence
- Add CSV export.
- Add one smoke-test run.

---

## 4) When you feel stuck

Run this again:

```bash
python start_here.py
```

Then send me:
1) the command you ran, and 2) the full error text.
I can then walk you through the *exact* next fix.
