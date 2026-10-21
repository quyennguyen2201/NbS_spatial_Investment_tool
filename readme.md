# Project Template Repository

This repository provides a **standardised project structure** for research projects.

It is designed to promote:

* Reproducibility
* Clear separation of inputs, outputs, and code
* Easier collaboration

---

## How to Use This Template

1. Click **“Use this template”** (top right of this repository).
2. Choose an owner (STRAND2)
3. Name your new project repository.
4. Create the repository.
5. Clone your new repo locally:

```bash
git clone https://github.com/<org>/<your-project>.git
cd <your-project>
```

## 📁 Repository Structure

```
project-name/
│
├── config/
│   ├── project_config.yml
│   └── user_config.yml
│
├── inputs/
│
├── notebooks/
│   └── notebook_stub.ipynb
│
├── lib/
│   └── helpers.py
│
├── outputs/
│
├── requirements.txt
├── .gitignore
│
└── README.md
```

---

## 📂 Folder & File Overview

### `config/`

Contains configuration files.

* **`project_config.yml`**
  Project-wide settings (paths, parameters, model settings).

* **`user_config.yml`**
  User-specific settings (local paths, environment variables, machine-specific paths).
  This file should not contain secrets or credentials.

---

### `inputs/`

For **small input datasets** required for the project.

Examples:

* CSV files
* Small GeoJSON files
* Lookup tables

⚠ Large datasets should not be committed to Git. Use backblaze instead

---

### `notebooks/`

Jupyter notebooks for:

* Exploration
* Analysis
* Prototyping
* Visualisation

Keep notebooks clean and reproducible.
Reusable logic can be moved into `lib/`.

#### Converting Notebooks to Python Scripts

When necessary, notebooks can be converted to Python scripts.

From the command line:

```bash
jupyter nbconvert --to script notebooks/analysis.ipynb
```

This will generate:

```
notebooks/analysis.py
```

---

### `lib/`

Reusable Python code.

* `helpers.py` contains shared utility functions.
* Additional modules can be added as needed.

---

### `outputs/`

For **small generated outputs**, such as:

* Processed CSV files
* Summary tables
* Plots
* Logs

⚠ Large outputs should not be committed to Git.

---

### `requirements.txt`

Lists the Python dependencies required to run the project.

After cloning the repository, install dependencies using:

```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### `.gitignore`

Specifies files and directories that should not be tracked by Git.

---

## 🔁 Recommended Workflow

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Create a new branch for your work:

```bash
git checkout -b my-branch-name
```

5. Update `project_config.yml` with project parameters.
6. Update `user_config.yml` with your local paths if necessary.
7. Create a .env file for storing credentials (see below)
8. Keep raw and large data outside Git (preferably backblaze).
9. Write reusable code in `lib/`.
10. Use notebooks for development and exploration.
11. Convert stable notebooks into Python scripts when necessary.
12. Commit changes **frequently**.
13. Push changes to github so that they're visible to others

---

### Storing secrets using a `.env` File

Create a `.env` file in the **project root directory**.

#### Step 1 — Create the file

From the project root create a new file named:

```
.env
```

The structure should look like:

```
project-name/
│
├── .env
├── config/
├── lib/
...
```

#### Step 2 — Add your secrets

Example `.env` contents:

```
B2_S3_ENDPOINT=https://s3.us-west-002.backblazeb2.com
B2_S3_ACCESS_KEY_ID= CONTACT GREG BODEKER
B2_S3_SECRET_ACCESS_KEY= CONTACT GREG BODEKER
STRAND_BUCKET=STRAND2
```
⚠ Do not wrap values in quotes unless necessary.

Make sure `.env` is included in `.gitignore`:

#### Step 3 — Load in Python

Then load the variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
```

If you are using the backblaze helper module these variables will be loaded automatically
