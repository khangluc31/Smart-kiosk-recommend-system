## Run

### 1. Prerequisites

Before running the project, make sure the following tools are installed:

- Python 3.10+ (with a compatible PyTorch environment)
- PostgreSQL
- Node.js and npm
- Git
- A Python virtual environment (recommended)

The project consists of three main components:

- **Database:** PostgreSQL for menu data, customer information, and purchase history.
- **AI Backend:** FastAPI, BERT NER, LinUCB, Two-Tower, LightGBM, Item2Vec, and ILP.
- **Frontend:** React and Vite.

### 2. Clone the Repository

```bash
git clone https://github.com/khangluc31/Smart-kiosk-recommend-system.git
cd Smart-kiosk-recommend-system
```

The root directory can be renamed. However, the commands below assume they are executed from the specified subdirectories.

### 3. Set Up the Python Environment

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python dependencies:

```bash
pip install numpy pandas scipy scikit-learn
pip install torch transformers accelerate
pip install lightgbm faiss-cpu
pip install fastapi uvicorn pydantic
pip install psycopg2-binary requests
pip install edge-tts pygame sounddevice faster-whisper
pip install pulp python-dotenv
```

Additional dependencies may be required depending on the environment and the current implementation.

For reproducibility, a pinned `requirements.txt` is recommended.

### 4. Set Up PostgreSQL

**Important: The database must be initialized before training the recommendation models.**

Create the PostgreSQL database:

```bash
psql -U postgres -d postgres
```

```sql
CREATE DATABASE recsys_db;
```

Exit PostgreSQL:

```sql
\q
```

Import the database schema:

```bash
psql -U postgres -d recsys_db -f Database/schema.sql
```

Update the database credentials in the relevant Python configuration files.

For example:

```python
DB_CONFIG = {
    "dbname": "recsys_db",
    "user": "postgres",
    "password": "YOUR_POSTGRESQL_PASSWORD",
    "host": "localhost",
    "port": "5432",
}
```

Replace `YOUR_POSTGRESQL_PASSWORD` with your own PostgreSQL password.

If environment variables are supported, using a local `.env` file is recommended instead of storing the password directly in the source code.

Next, populate the database:

```bash
cd Database

python3 menu.py
python3 dummy.py
```

- `menu.py`: Retrieves KFC menu data through the KFC Vietnam menu API and inserts it into PostgreSQL.
- `dummy.py`: Generates synthetic user profiles and purchase history for development and testing.

**Note:** The generated purchase history is synthetic and should not be treated as real customer behavior.

Return to the project root:

```bash
cd ..
```

### 5. Train the BERT NER Model

BERT NER is trained separately from the recommendation models.

Its purpose is to identify:

- `want`: Food items customers want to order.
- `unwant`: Food items or ingredients customers want to avoid.

The model uses BIO token classification with five labels:

```python
LABELS = {
    0: "O",
    1: "B-want",
    2: "B-unwant",
    3: "I-want",
    4: "I-unwant",
}
```

The training datasets are stored in the `Train/` directory.

To train the model:

1. Collect the relevant JSONL training files from `Train/`.
2. Upload the training script, datasets, and required supporting modules (including `context.py` when needed) to Kaggle or Google Colab, or train locally.
3. Train the BERT NER model using the combined training dataset.
4. Download the resulting model directory.

Example local command:

```bash
cd AI

python3 train_ner.py \
    ../Train/your_training_dataset.jsonl \
    --out models/ner_model \
    --epochs 4
```

Multiple training datasets can be provided as separate positional arguments if supported by the trainer.

The expected output is:

```text
AI/
└── models/
    └── ner_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer_config.json
        └── ...
```

The model is based on `bert-base-multilingual-cased` and evaluated using entity-level Precision, Recall, and F1-score.

### 6. Train the Recommendation Models

After populating PostgreSQL, train the recommendation models.

From the `AI/` directory:

```bash
python3 train.py
```

This script trains and evaluates:

| Model | Purpose | Evaluation |
|---|---|---|
| LinUCB | Recommendations for new customers | Recall@5, Recall@20 |
| Two-Tower | Personalized retrieval for returning customers | Recall@5, Recall@20 |
| LightGBM | Ranking retrieved candidates | Recall@5 |
| Item2Vec | Complementary item recommendations | Recall@1, Recall@3 |

The models are saved in the `models/` directory.

Expected structure:

```text
AI/
└── models/
    ├── ner_model/
    ├── linucb.pkl
    ├── two_tower.pt
    ├── ranker.txt
    └── item2vec.pkl
```

If BERT NER was trained on Kaggle or Google Colab, copy its exported `ner_model/` directory into `AI/models/`.

**Important:** If the database is rebuilt or meal IDs change, retraining the recommendation models may be necessary to ensure compatibility between the trained models and the current database.

### 7. Run the FastAPI Backend

From the `AI/` directory, start the backend:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Make sure PostgreSQL is running and that the trained model files are in their expected locations.

### 8. Run the React Frontend

Open another terminal.

From the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

```text
http://localhost:5173
```

The frontend communicates with the FastAPI backend through the `/api` endpoints.

If required, configure the backend URL in `frontend/.env.local`:

```dotenv
VITE_API_URL=http://127.0.0.1:8000
```

Do not place database passwords or other secrets in frontend environment variables.

### 9. Important Notes

- PostgreSQL must be initialized before training the recommendation models.
- BERT NER can be trained independently of the recommendation models.
- Run the training and backend commands from the `AI/` directory unless paths are configured otherwise.
- Update PostgreSQL credentials according to your local environment.
- Do not commit `.env` files, virtual environments, or `node_modules/`.
- Synthetic training data is intended for development and experimentation.
- Offline evaluation metrics do not necessarily represent real-world recommendation performance.

## Disclaimer

The user profiles, customer names, and purchase histories contained in the synthetic training data were generated using large language models and predefined simulation rules.

These records do not represent real individuals or actual KFC customers. Any resemblance to real names, individuals, identities, or purchasing behavior is purely coincidental.

This project is developed for academic, educational, and demonstration purposes and is not an official KFC product.