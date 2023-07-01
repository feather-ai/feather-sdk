include .local/secrets.env
ifeq ($(OS),Windows_NT)
export PYTHONPATH=$(CURDIR)/src;$(CURDIR)/src/feather
else
export PYTHONPATH=$(shell pwd)/src:$(shell pwd)/src/feather
endif

init:
	pip3 install -r src/requirements.txt

tests:
	cd src/unit_tests && pytest -q

vqg:
	FEATHER_PUBLISH_BASE_URL=http://localhost:8080 python3 src/examples/vqg/ftr.py

example1:
	FEATHER_PUBLISH_BASE_URL=http://localhost:8080 python3 src/examples/example1.py

example2:
	FEATHER_PUBLISH_BASE_URL=http://localhost:8080 python3 src/examples/example2.py

example1_cloud:
	python3 src/examples/example1.py

example2_cloud:
	python3 src/examples/example2.py

faitoy:
	cd src/examples/fai-toy-model && python3 inference.py

publish:
	docker build --build-arg PYPI_USERNAME=$(FEATHER_PYPI_USERNAME) --build-arg PYPI_PASSWORD=$(FEATHER_PYPI_PASSWORD) . -t feather-sdk-package

html:
	rm -fr src/feather/featherlocal/www/public
	cd src/feather/featherlocal/www/node && npm run build