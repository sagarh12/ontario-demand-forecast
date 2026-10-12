PY ?= python

.PHONY: install data train evaluate test app all

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

test:
	$(PY) -m pytest -q

app:
	streamlit run app.py

all: data train evaluate
