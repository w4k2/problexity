.PHONY: all clean test

profile:
	kernprof -l -v problexity/tests/test_common.py

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf doc/_build
	rm -rf doc/auto_examples
	rm -rf doc/generated
	rm -rf doc/modules
	rm -rf examples/.ipynb_checkpoints

docs: clean install
	cp -rf ./plots ./doc/
	cp -rf ./examples/*.png ./doc/_static
	cd doc && make html
	#cd doc && make latex

test-code:
	py.test problexity

test-coverage:
	rm -rf coverage .coverage
	py.test --cov-report term-missing:skip-covered --cov=problexity problexity

test: clean test-coverage

run: clean
	python workspace.py

code-analysis:
	flake8 problexity | grep -v __init__
	pylint -E problexity/ -d E1103,E0611,E1101

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*
	pip3 install --upgrade problexity

install: clean
	python setup.py clean
	python setup.py develop
