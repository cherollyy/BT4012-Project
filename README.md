# BT4012-Project

# 🛡️ FraudGuard AI Dashboard

A comprehensive fraud detection system with interactive data visualizations and real-time fraud prediction capabilities. Built with Streamlit (frontend) and FastAPI (backend), containerized with Docker.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Frontend Features](#frontend-features)
- [Backend Features](#backend-features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

FraudGuard AI is a machine learning-powered fraud detection dashboard that helps identify fraudulent e-commerce transactions. The system consists of:

- **Frontend**: Interactive Streamlit dashboard for data visualization and fraud checking
- **Backend**: FastAPI service with ML model training and prediction endpoints
- **ML Model**: Ensemble predictor for fraud classification

---

## ✨ Features

### Frontend (Streamlit Dashboard)

#### 📊 Dashboard Tab
- **Key Metrics**: Total transactions, fraud count, fraud rate, average transaction amount
- **Transaction Amount Distribution**: Interactive histogram with log scale and sampling controls
- **Fraud vs Legit Classification**: Visual comparison with color-coded bars (red = fraud, green = legit)
- **Fraud Rate Analysis**: By payment method, device type, product category
- **Time Series Analysis**: Daily transaction trends with 7-day moving average
- **Fraud Heatmap**: Hour-of-day vs day-of-week visualization
- **Customer Demographics**: Age distribution in fraud cases
- **Outlier Detection**: Scatter plot of transaction amount vs account age
- **IP-coded choropleth graph**: cloropleth graph showing frequency of fraud cases

#### 🔍 Fraud Checker Tab
- **User Fraud Check**: Analyze fraud risk based on user profile (age, account age, order history)
- **Transaction Fraud Check**: Analyze individual transactions (amount, payment method, device, IP, browser)
- **Advanced Options**: Historical aggregate proxies for improved prediction accuracy
- **Real-time Predictions**: Get fraud probability percentage 

### Backend (FastAPI)

- **Model Training**: Automatic or manual ML model training
- **Fraud Prediction**: RESTful API for fraud probability scoring
- **Status Endpoints**: Check model loading status and training progress
- **Persistent Storage**: Trained models saved for quick startup
- **Background Training**: Non-blocking model training with progress tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                              │
│                 http://localhost:8501                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Frontend Container (Streamlit)                  │
│  - Data visualization (Altair, Plotly, Seaborn)            │
│  - Interactive UI components                                 │
│  - Communicates with backend API                            │
│  Port: 8501                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend Container (FastAPI)                     │
│  - ML model training & inference                            │
│  - RESTful API endpoints                                    │
│  - Model persistence                                         │
│  Port: 8000                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Persistent Storage                         │
│  - Training data (CSV)                                       │
│  - Trained models (.pkl)                                    │
│  - GeoIP database (optional)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

- **Docker Desktop** (Windows/Mac/Linux)
- **Git** (for cloning the repository)
- At least **4GB RAM** available for Docker
- **Python 3.10+** (if running locally without Docker)

### System Requirements

- **Windows**: Windows 10/11 (64-bit) with WSL 2 enabled
- **macOS**: macOS 10.15 or newer
- **Linux**: Any modern distribution with Docker support

---

## 🚀 Installation

### Step 1: Ensure all files are present
Pull all files from github 

### Step 2: Prepare Data

Place your training data in the `data/` folder:

```
data/
├── Fraudulent_E-Commerce_Transaction_Data_2.csv
```

### Step 3: Install Docker Desktop (important)

If you don't have Docker installed:

1. Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Install and restart your computer
3. Start Docker Desktop and wait for it to be ready

### Step 4: Verify Docker Installation

```bash
docker --version
docker compose version
```

---

## 🎮 Usage

### Starting the Application

From the project root directory:

```bash
# Build and start both frontend and backend
docker compose up --build

# Or run in detached mode (background)
docker compose up --build -d
```

**First-time startup**: The backend will train the ML model automatically. This takes **30 seconds to 5 minutes** depending on your system.

**Subsequent startups**: If the trained model exists, startup is instant (~2 seconds).

### Accessing the Dashboard

Once running, open your browser and navigate to:

- **Frontend Dashboard**: http://localhost:8501 (paste this into browser, recommended)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (interactive Swagger UI)

### Stopping the Application

```bash
# If running in attached mode (logs showing)
# Press Ctrl+C, then run:
docker compose down

# If running in detached mode
docker compose down
```

### Viewing Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f frontend
docker compose logs -f backend
```

---

## 🎨 Frontend Features

### Dashboard Page

#### 1. Key Performance Indicators (KPIs)
- Total transactions count
- Total fraud cases
- Fraud rate percentage
- Average transaction amount

#### 2. Interactive Visualizations

**Transaction Amount Distribution**
- Adjustable histogram with bin control (10–200 bins)
- Log scale toggle for better visualization of skewed data
- Sample size slider to improve performance (0–5000 rows)
- Max amount filter to focus on specific ranges

**Fraud Classification**
- Color-coded bar chart (red = fraud, green = legit)
- Shows class imbalance clearly

**Fraud Rate by Payment Method**
- Compare fraud rates across Credit Card, Debit Card, PayPal, Bank Transfer
- Identify high-risk payment methods

**Daily Transaction Trends**
- Time series plot with 7-day moving average
- Identify seasonal patterns and anomalies

**Fraud Heatmap**
- Hour-of-day (0–23) vs Day-of-week (Mon–Sun)
- Identify peak fraud times

**Product Category Analysis**
- Top 20 categories with highest fraud counts
- Horizontal bar chart sorted by frequency

**Customer Age Distribution**
- Age distribution among fraud cases
- Identify high-risk age groups

**Outlier Detection Scatter Plot**
- Transaction Amount vs Account Age
- Color-coded by fraud status
- Sample-based for performance (2000 rows)

**IP-coded choropleth graph**
- cloropleth graph showing frequency of fraud cases
- uses ip-address column in data

  
### Fraud Checker Page

#### User Fraud Check
Input fields:
- Age (18–100)
- Account Age (days)
- Past Fraud Cases
- Average Order Value
- Total Transactions
- **Advanced options**: Customer transaction history, IP count

#### Transaction Fraud Check
Input fields:
- Transaction Amount
- Payment Method
- Device Type
- IP Address
- Browser Type
- Shipping Address
- Billing Address
- Transaction Hour (0–23)
- Customer Age
- **Advanced options**: Amount per unit, customer totals, IP transaction count

**Output**: Fraud probability percentage (0–100%)

---

## 🔧 Backend Features

### API Endpoints

#### `GET /model_status`
Check if the ML model is loaded and ready.

**Response:**
```json
{
  "loaded": true,
  "metadata": {
    "feature_names": [...],
    "model_type": "EnsemblePredictor",
    "trained_at": "2025-01-21T14:30:00"
  },
  "message": "Model loaded successfully"
}
```

#### `GET /train_status`
Get current training progress.

**Response:**
```json
{
  "in_progress": true,
  "percent": 65,
  "message": "Training in progress (65%)"
}
```

#### `POST /train`
Trigger model training manually.

**Response:**
```json
{
  "status": "started",
  "message": "Training started in background"
}
```

#### `POST /predict_fraud`
Get fraud probability for a transaction or user.

**Request (User check):**
```json
{
  "features": {
    "age": 30,
    "account_age_days": 365,
    "avg_order_value": 150.0,
    "total_transactions": 15
  }
}
```

**Request (Transaction check):**
```json
{
  "transaction": {
    "amount": 299.99,
    "payment_method": "Credit Card",
    "device": "Mobile",
    "ip": "192.168.1.1",
    "transaction_hour": 14,
    "age": 30
  }
}
```

**Response:**
```json
{
  "fraud_probability": 23.5,
  "prediction": "Legit",
  "confidence": 76.5
}
```

### Model Training

The backend uses an ensemble ML model that combines:
- Gradient Boosting
- Random Forest
- Logistic Regression

**Training is triggered:**
1. Automatically on first startup (if no model exists)
2. Manually via `/train` endpoint
3. Manually by running `train_model.py` inside the container

**Model persistence:**
- Trained models saved to `/app/models/fraud_model.pkl`
- Persisted via Docker volume mount to `./backend/models/`
- Subsequent startups load existing model (no retraining needed)

---

## ⚙️ Configuration

### Environment Variables

Set these in `docker-compose.yml`:

**Frontend:**
```yaml
environment:
  - API_URL=http://backend:8000          # Backend API endpoint
  - MODEL_WAIT_TIMEOUT=120               # Max wait time for model (seconds)
```

**Backend:**
```yaml
environment:
  - PYTHONUNBUFFERED=1                   # Python logging
  - RAW_DATA_PATH=/app/data/Fraudulent_E-Commerce_Transaction_Data_2.csv
  - MODEL_PATH=/app/models/fraud_model.pkl
```

### Port Configuration

Change ports in `docker-compose.yml` if 8501 or 8000 are already in use:

```yaml
services:
  frontend:
    ports:
      - "8502:8501"  # Host:Container (change 8502 to your preferred port)
  
  backend:
    ports:
      - "8001:8000"  # Host:Container
```

Then update `API_URL` in frontend environment to match.

### Volume Mounts

Persist data and models across container restarts:

```yaml
services:
  backend:
    volumes:
      - ./data:/app/data              # Training data
      - ./backend/models:/app/models  # Trained models (persisted)
  
  frontend:
    volumes:
      - ./data:/app/data              # Read-only access to data
```

---

## 🐛 Troubleshooting

### "Model not loaded" Error

**Symptom**: Fraud Checker shows "Model still not loaded after waiting"

**Solutions:** 
1. **Wait for training to complete** (first startup takes 30s–5min): go to console and check that a fraud model pkl file was saved. If not, wait. 
2. **Click "Retry model status"** after training completes: Once the pkl is saved, go back to the dashboard and click retry model
3. **Manually trigger training**:
   ```bash
   curl http://localhost:8000/train
   ```
4. **Check backend logs**:
   ```bash
   docker compose logs backend
   ```
5. **Verify data file exists**:
   ```bash
   docker exec backend-1 ls -la /app/data
   ```

### "Data file not found" Error

**Symptom**: Dashboard shows "❌ Data file not found"

**Solutions:**
1. Ensure CSV file exists in `data/` folder
2. Check filename matches exactly: `Fraudulent_E-Commerce_Transaction_Data_2.csv`
3. Verify volume mount in `docker-compose.yml`
4. Check file permissions (should be readable)

### Port Already in Use

**Symptom**: `Error: bind: address already in use`

**Solutions:**
1. **Stop existing containers**:
   ```bash
   docker compose down
   ```
2. **Find and kill process using the port** (Windows PowerShell):
   ```powershell
   Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process
   ```
3. **Change port** in `docker-compose.yml` (see Configuration section)

### Cannot Connect to Backend

**Symptom**: Frontend shows connection errors

**Solutions:**
1. **Verify backend is running**:
   ```bash
   docker ps
   ```
   Should show both `frontend-1` and `backend-1`

2. **Check backend logs**:
   ```bash
   docker compose logs backend
   ```

3. **Verify API_URL** in docker-compose.yml:
   ```yaml
   environment:
     - API_URL=http://backend:8000  # Must match backend service name
   ```

4. **Test backend directly**:
   ```bash
   curl http://localhost:8000/
   ```

### Slow Performance

**Solutions:**
1. **Reduce sample size** in visualizations (use sliders)
2. **Allocate more RAM to Docker**:
   - Docker Desktop → Settings → Resources → Memory (increase to 4GB+)
3. **Use detached mode**:
   ```bash
   docker compose up -d
   ```

### Fresh Restart

If things are broken, start fresh:

```bash
# Stop and remove everything
docker compose down -v

# Remove old models
rm -rf backend/models/*.pkl

# Rebuild from scratch
docker compose build --no-cache
docker compose up
```

---

## 🔄 Development Workflow

### Making Code Changes

**Frontend changes** (app.py):
1. Edit `frontend/app.py`
2. Save the file
3. Streamlit will auto-reload (if using volume mount)
4. Refresh browser

**Backend changes** (main.py):
1. Edit `backend/main.py`
2. Restart backend:
   ```bash
   docker compose restart backend
   ```

**Dependency changes** (requirements.txt):
1. Edit requirements.txt
2. Rebuild:
   ```bash
   docker compose build
   docker compose up
   ```

### Running Locally (Without Docker)

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---



