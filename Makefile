SHELL := /bin/bash

# setup
ifdef VIRTUAL_ENV
PYTHON := $(VIRTUAL_ENV)/bin/python
PIP := $(VIRTUAL_ENV)/bin/pip
endif

# Create a local virtual environment
# be sure to source venv/bin/activate after this!
.PHONY: env
env :
	python3.12 -m venv venv

# you need to have sourced the venv first
.PHONY: install-requirements
install-requirements:
	$(PIP) install -r requirements.txt

.PHONY: install-dev-requirements
install-dev-requirements:
	$(PIP) install mypy yapf

# patch borked python libs
.PHONY: patch
patch :
	patch -u venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py -i patches/mujoco_rendering.py.patch

# you need to have sourced the venv first
.PHONY: format
format:
	yapf -i --recursive  GTS testing train.py play.py paramSearch.py info.py

# you need to have sourced the venv first
.PHONY: lint
lint:
	mypy GTS/*.py testing train.py play.py paramSearch.py info.py

# you need to have sourced the venv first
.PHONY: test
test:
	pytest testing -rS -v
