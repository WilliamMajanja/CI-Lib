.PHONY: install test run-backend run-frontend docker-up docker-down lint

install:
	pip install -e ".[dev]"
	pip install fastapi uvicorn pydantic streamlit matplotlib

test:
	python -m pytest tests/ -v

run-backend:
	python -m backend.app

run-frontend:
	streamlit run frontend/app.py --server.port=8501

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

lint:
	python -m pip install ruff
	python -m ruff check ci_lib/ backend/ frontend/
