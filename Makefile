.PHONY: install data train evaluate api dashboard docker clean all

install:
	pip install -r requirements.txt

data:
	python -m src.data.generate_data

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

docker:
	docker-compose up --build

clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/models/*.pkl plots/
	find . -type d -name __pycache__ -exec rm -rf {} +

all: install data train evaluate
