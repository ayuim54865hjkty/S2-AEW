.PHONY: test demo lint clean

test:
	python -m compileall s2aew examples tests
	python -m pytest tests -q

demo:
	python examples/toy_example.py

lint:
	python -m compileall s2aew examples tests

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
