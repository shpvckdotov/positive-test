# Log Clustering Service

Сервис автоматической кластеризации событий безопасности для аналитиков SOC.  
Группирует Windows Event Logs, Linux syslog и HDFS-логи, выявляет аномалии и предоставляет REST API + Web UI.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                         Docker Compose                       │
│                                                             │
│  ┌──────────────────────────────┐   ┌──────────────────┐   │
│  │         FastAPI App           │   │   PostgreSQL 16   │   │
│  │                              │   │                  │   │
│  │  ┌──────────────────────┐   │   │  log_events      │   │
│  │  │  LogPreprocessor     │   │◄──►│  clusters        │   │
│  │  │  (parser + normalize)│   │   │                  │   │
│  │  └──────────┬───────────┘   │   └──────────────────┘   │
│  │             │               │                           │
│  │  ┌──────────▼───────────┐   │                           │
│  │  │  LogClusteringService│   │                           │
│  │  │  TF-IDF + DBSCAN     │   │                           │
│  │  │  (offline trained)   │   │                           │
│  │  └──────────────────────┘   │                           │
│  │                              │                           │
│  │  REST API  /api/v1/…        │                           │
│  │  Web UI    /  /clusters-view│                           │
│  │           /visualize-view   │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Компоненты

| Компонент | Описание |
|---|---|
| `app/services/preprocessing.py` | Парсинг и нормализация логов: regex-шаблоны для Linux/Windows/HDFS, замена IP → `<IP>`, чисел → `<NUM>` и т.д. |
| `app/services/clustering.py` | TF-IDF (bigram, 5000 признаков) + DBSCAN (cosine distance). Offline-обучение, online-инференс через nearest-centroid. |
| `app/services/log_store.py` | Async-репозиторий (SQLAlchemy 2.0) для хранения событий и кластеров в PostgreSQL. |
| `app/api/routes.py` | REST API: классификация, просмотр кластеров/событий, статистика, t-SNE визуализация. |
| `app/templates/` | Web UI: Jinja2 + Chart.js (scatter plot), без сборщика — один файл CSS. |
| `ml/train.py` | Оффлайн-скрипт обучения модели. |
| `ml/evaluate.py` | Скрипт оценки качества кластеризации. |
| `data/download_data.py` | Генерация синтетических или загрузка реальных LogHub-данных. |

---

## Технологии и обоснование выбора

### ML-пайплайн

**Предобработка — Drain-style нормализация**  
Логи содержат переменные части (IP, PID, имена файлов), которые мешают кластеризации.  
Заменяем их токенами (`<IP>`, `<NUM>`, `<PATH>`, `<UUID>`, `<HEX>`) с помощью regex — это стандартный подход.

**Векторизация — TF-IDF (1+2-gram)**  
- Быстро, интерпретируемо, хорошо работает на шаблонных текстах (логи ≠ свободный текст).  
- Sublinear TF (`1 + log(tf)`) снижает доминирование частых токенов.  
- Альтернатива — sentence-transformers, но это избыточно для структурированных логов.

**Кластеризация — DBSCAN (cosine distance)**  
- Не требует задавать число кластеров заранее.  
- Точки с низкой плотностью (label = -1) автоматически становятся **аномалиями** — это ключевое для SOC.  
- Cosine distance инвариантна к длине документа.  
- Альтернативы: K-Means (нужен k), Agglomerative (медленно на больших данных).

**Инференс новых событий — nearest centroid**  
После обучения каждый кластер представлен L2-нормированным центроидом TF-IDF.  
Новый лог → TF-IDF → cosine similarity → ближайший кластер.  
Если similarity < порога — событие помечается как аномалия (без переобучения).

### Backend
- **FastAPI** — async, OpenAPI из коробки, pydantic-валидация.  
- **SQLAlchemy 2.0 async** + **asyncpg** — неблокирующий I/O для PostgreSQL.  
- **PostgreSQL** — хранение всех событий и кластеров с индексами.

### Frontend
- Jinja2 + Chart.js (CDN) — без сборщика, минимальные зависимости.
- t-SNE (sklearn) → интерактивный scatter plot.

---

## Быстрый старт (Docker)

```bash
git clone <repo-url>
cd log-clustering-service

# Запуск (PostgreSQL + FastAPI, обучение модели встроено в образ)
docker compose up --build

# Сервис будет доступен на:
#   Web UI:  http://localhost:8000
#   API:     http://localhost:8000/api/v1/
#   Swagger: http://localhost:8000/docs
```

> Первый `docker compose up --build` занимает ~3–5 минут (установка зависимостей + обучение модели на синтетических данных).

---

## Локальный запуск (без Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# PostgreSQL должен быть запущен
cp .env.example .env
# Отредактируйте DATABASE_URL в .env

# 1. Сгенерировать данные
python data/data_download.py

# 2. Обучить модель
python ml/train.py

# 3. Оценить качество
python ml/evaluate.py

# 4. Запустить сервисы
docker-compose up --build 

```

# В браузере ввести url [localhost:8000](http://localhost:8000)

### Использование реальных данных LogHub

---

## API — примеры запросов

### POST /api/v1/classify — классифицировать событие

```bash
# Linux syslog — brute force SSH
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "raw_log": "Jun 14 15:16:01 combo sshd[19939]: Failed password for root from 218.188.2.4 port 22 ssh2",
    "source_type": "linux"
  }'
```

```json
{
  "event": {
    "id": 1,
    "source_type": "linux",
    "log_level": null,
    "hostname": "combo",
    "process_name": "sshd",
    "message": "Failed password for root from 218.188.2.4 port 22 ssh2",
    "cluster_id": 3,
    "is_anomaly": false,
    "confidence": 0.87,
    "created_at": "2026-04-06T12:00:00"
  },
  "cluster": {
    "id": 3,
    "label": 2,
    "template": "Failed password for root from 218.188.2.4 port 22 ssh2",
    "source_type": "linux",
    "size": 142
  },
  "is_anomaly": false,
  "confidence": 0.87,
  "message": "Assigned to cluster #2 with confidence 87.00%."
}
```

```bash
# Windows Event Log — успешный вход
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "raw_log": "2004-11-04 03:26:32, Information, 4624, Security, An account was successfully logged on. Subject: Security ID: S-1-5-18 Logon Type: 3",
    "source_type": "windows"
  }'
```

```bash
# HDFS — DataNode block receive
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "raw_log": "081109 204005 1 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.0.0.1:54106 dest: /10.0.0.2:50010",
    "source_type": "hdfs"
  }'
```

### GET /api/v1/clusters — список кластеров

```bash
curl http://localhost:8000/api/v1/clusters
```

### GET /api/v1/clusters/{id} — детали кластера

```bash
curl http://localhost:8000/api/v1/clusters/3
```

### GET /api/v1/events — последние события

```bash
# Все события (50 шт.)
curl http://localhost:8000/api/v1/events

# Только Linux, 100 событий
curl "http://localhost:8000/api/v1/events?limit=100&source_type=linux"

# Только аномалии (cluster_id кластера аномалий)
curl "http://localhost:8000/api/v1/events?cluster_id=1"
```

### GET /api/v1/stats — статистика

```bash
curl http://localhost:8000/api/v1/stats
```

```json
{
  "total_events": 1523,
  "total_clusters": 12,
  "anomaly_count": 47,
  "anomaly_rate": 0.031,
  "by_source": {"linux": 800, "windows": 600, "hdfs": 123},
  "top_clusters": [...]
}
```

### GET /api/v1/visualize — данные для t-SNE визуализации

```bash
curl "http://localhost:8000/api/v1/visualize?limit=500"
```

### POST /api/v1/retrain — переобучить модель на накопленных данных

```bash
curl -X POST http://localhost:8000/api/v1/retrain
```

---

## Тесты

```bash
pip install pytest
pytest tests/ -v
```

---

## Оценка качества кластеризации

Результаты на синтетическом датасете :

| Метрика | Значение | Интерпретация |
|---|---|---|
| Число кластеров | ~80 | Соответствует числу шаблонов |
| Noise ratio | ~5–15% | Аномальные/редкие события |
| Silhouette score | 0.7–0.82 | Хорошее разделение кластеров (диапазон −1..1) |
| Davies-Bouldin | 0.7-0.8 | Компактные, хорошо разделённые кластеры (ниже = лучше) |


### Как запустить оценку вручную

```bash
python ml/evaluate.py 
```

---

## Структура проекта

```
.
├── app/
│   ├── api/routes.py          # REST API эндпоинты
│   ├── core/config.py         # Настройки (pydantic-settings)
│   ├── db/
│   │   ├── database.py        # Async SQLAlchemy engine
│   │   └── models.py          # ORM-модели (Cluster, LogEvent)
│   ├── schemas/log_schema.py  # Pydantic схемы запросов/ответов
│   ├── services/
│   │   ├── clustering.py      # TF-IDF + DBSCAN + сохранение модели
│   │   ├── log_store.py       # Async репозиторий БД
│   │   └── preprocessing.py   # Парсинг и нормализация логов
│   ├── static/                # CSS, JS
│   ├── templates/             # Jinja2 HTML
│   └── main.py                # FastAPI приложение
├── ml/
│   ├── train.py               # Обучение модели
│   └── evaluate.py            # Оценка качества
├── data/
│   └── download_data.py       # Генерация/загрузка данных
├── tests/
│   ├── test_preprocessing.py
│   └── test_clustering.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
