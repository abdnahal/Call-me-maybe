clean:
	rm -rf __pycache__/
	rm *.py[oc]
	rm -rf build/
	rm -rf dist/
	rm -rf wheels/
	rm *.egg-info
debug:
	python -m pdb -m src
install:
	uv sync

run:
	uv run python -m src

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
	  --ignore-missing-imports --disallow-untyped-defs \
	  --check-untyped-defs
