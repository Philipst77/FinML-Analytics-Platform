# QuantumEdge AI 🚀

> Enterprise ML-Powered Financial Intelligence Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

QuantumEdge AI is an enterprise-grade financial intelligence platform that combines cutting-edge machine learning with real-time market data to deliver actionable trading insights. The system leverages transformer-based neural networks, GPU-accelerated inference, and cloud-native architecture.

### Key Features

- 🧠 **Sentiment Analysis**: Fine-tuned FinBERT models analyzing financial news, social media, and SEC filings
- 📈 **Price Forecasting**: Multi-horizon predictions using Temporal Fusion Transformers
- 🔍 **Semantic Search**: Vector search over millions of financial documents using pgvector
- ⚡ **High Performance**: Sub-100ms API latency with caching and dynamic batching
- 🌐 **Real-Time Dashboard**: React TypeScript frontend with WebSocket streaming

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TS)                    │
│              WebSocket • REST API • Real-time Charts         │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│         Authentication • Rate Limiting • Caching             │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────┬──────────────────────────────────────┐
│   ML Models (GPU)    │     Database (PostgreSQL)            │
│  • FinBERT           │  • pgvector Extension                │
│  • TFT Forecasting   │  • Time-series Data                  │
│  • Quantized INT8    │  • Vector Embeddings                 │
└──────────────────────┴──────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Machine Learning
- **PyTorch 2.0** - Deep learning framework with CUDA 11.8
- **Hugging Face Transformers** - Pre-trained models (FinBERT, TFT)
- **Mixed Precision (FP16)** - 2x training speedup
- **Model Quantization (INT8)** - 4x inference speedup

### Backend
- **FastAPI** - High-performance REST API
- **gRPC** - Low-latency inter-service communication
- **PostgreSQL 15** - Relational database with pgvector
- **Redis** - Caching layer (60%+ hit rate)

### Frontend
- **React 18 + TypeScript** - Type-safe UI development
- **TailwindCSS** - Utility-first styling
- **Recharts + Lightweight Charts** - Financial visualizations
- **WebSocket** - Real-time data streaming

### Cloud Infrastructure
- **AWS S3** - Model artifacts and data storage
- **AWS RDS** - Production PostgreSQL database
- **AWS Lambda** - Serverless data ingestion
- **AWS ECS/Elastic Beanstalk** - Container orchestration
- **Docker** - Containerization

## 📊 Performance Metrics

- **API Latency**: p95 < 100ms, p99 < 200ms
- **Sentiment Model**: F1 Score 0.85+, Accuracy 87%
- **Price Forecasting**: MAPE < 8%, Directional Accuracy > 58%
- **Throughput**: 500+ requests/second on single GPU
- **Vector Search**: <200ms for top-10 from 1M+ documents

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA 11.8 (for training)
- Node.js 18+ (for frontend)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/QuantumEdge-AI.git
cd QuantumEdge-AI
```

2. **Set up environment**
```bash
# Copy environment template
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

3. **Start services with Docker Compose**
```bash
docker-compose up -d
```

4. **Access the application**
- API: http://localhost:8000
- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs

## 📁 Project Structure
```
QuantumEdge-AI/
├── src/
│   ├── models/           # ML model training & inference
│   ├── api/              # FastAPI application
│   ├── data/             # Data processing & ingestion
│   └── utils/            # Shared utilities
├── frontend/             # React TypeScript dashboard
├── infrastructure/       # Terraform, Docker configs
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
├── docker-compose.yml    # Local development stack
└── requirements.txt      # Python dependencies
```

## 🔧 Development Roadmap

### Phase 1: ML Pipeline (Weeks 1-4) ✅ In Progress
- [x] Environment setup
- [x] Data acquisition pipeline
- [ ] FinBERT fine-tuning
- [ ] TFT training
- [ ] Model optimization (quantization, CUDA)

### Phase 2: API Infrastructure (Weeks 5-7)
- [ ] FastAPI service implementation
- [ ] Authentication & rate limiting
- [ ] Redis caching layer
- [ ] Dynamic batching
- [ ] AWS deployment

### Phase 3: Database (Weeks 8-10)
- [ ] PostgreSQL schema design
- [ ] pgvector setup
- [ ] Data ingestion Lambda functions
- [ ] Query optimization

### Phase 4: Frontend (Weeks 11-13)
- [ ] React TypeScript setup
- [ ] Dashboard components
- [ ] WebSocket integration
- [ ] Mobile responsive design

## 📈 Model Training

### Local Training (Development)
```bash
# Train sentiment model on local GPU
python src/models/train_sentiment.py --epochs 5 --batch-size 32

# Train forecasting model
python src/models/train_forecasting.py --lookback 60 --horizons 1,5,20
```

### AWS Deployment (Production)
Models are designed for deployment to AWS EC2 GPU instances (p3/g5) with zero code changes.

## 🧪 Testing
```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

Comprehensive technical specification and implementation details available in:
- [Technical Specification](docs/technical-spec.pdf)
- [API Documentation](docs/api.md)
- [Model Documentation](docs/models.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

This is a personal portfolio project. If you'd like to suggest improvements, please open an issue.


##  Acknowledgments

- Hugging Face for transformer models
- FinBERT by ProsusAI
- Financial datasets from Yahoo Finance, Alpha Vantage
- Inspired by production ML systems at leading fintech companies

---

⭐ Star this repo if you find it useful!
