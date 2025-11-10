PY=python3
VENV=./venv


.PHONY: venv install run format


venv:
$(PY) -m venv $(VENV)
. $(VENV)/bin/activate && pip install --upgrade pip


install: venv
. $(VENV)/bin/activate && pip install -r requirements.txt


run:
. $(VENV)/bin/activate && $(PY) -m kirkmesh.cli --src data/samples/input.jpg --tgt data/target_face.jpg --out outputs/result.jpg


format:
. $(VENV)/bin/activate && python -m black src