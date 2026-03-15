.PHONY: setup preprocess train index eval serve \
        docker-up docker-down test lint clean

setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

preprocess:
	python data/preprocess.py

train:
	python training/train.py --config training/config.yaml

index:
	python rag/indexer.py --docs_path ./data/raw

eval:
	python eval/run_eval.py

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

test:
	pytest tests/ -v --tb=short

lint:
	flake8 . --max-line-length=100 --exclude=checkpoints,data

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
