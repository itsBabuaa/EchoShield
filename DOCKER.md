# Docker Guide

## Quick Start

```bash
docker-compose up -d
```

Access at: http://localhost:5000

## Commands

**Start:**
```bash
docker-compose up -d
```

**Stop:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

**Rebuild:**
```bash
docker-compose up -d --build
```

## Environment Variables

Set in `.env` file or docker-compose.yml:

- `PORT` - Server port (default: 5000)
- `SECRET_KEY` - Flask secret key
- `GROQ_API_KEY` - Groq API key for chatbot (optional)

## Production Server

The Docker container uses **Gunicorn** (production WSGI server) with:
- 4 worker processes
- 120 second timeout
- Automatic request logging

## Requirements

- Docker installed
- Model file at `models/bilstm_model.keras`

That's it!
