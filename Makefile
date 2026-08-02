run:
	uv run python -m src

clean:
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf .mypy*
	rm -rf build/
	rm -rf dist/
	rm -rf wheels/

debug:
	python -m pdb -m src

install:
	uv sync

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores \
	  --ignore-missing-imports --disallow-untyped-defs \
	  --check-untyped-defs
