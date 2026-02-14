.PHONY: serve stop status logs list install dev

MODEL ?= qwen2.5-coder-32b-gptq

serve:
	llm-inference serve $(MODEL)

stop:
	llm-inference stop

status:
	llm-inference status

logs:
	llm-inference logs

list:
	llm-inference list

install:
	python3 -m venv .venv
	.venv/bin/pip install -e .

dev:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest
