# REPLICATION

## End-to-end reproduction

With access to the data, you should be able to reproduce all results and visualizations with:

    $ OPENBLAS_NUM_THREADS=1 METAOD_SERVICE_URL=http://hostname:port poetry run dvc repro

where `http://hostname:port` is the base URL of the MetaOD service launched under Python 3.7.

We assume that `python3` points to the Python 3.7 executable with the relevant MetaOD dependencies
available (see below).

### MetaOD service

MetaOD depends on outdated software versions. To integrate it with our evaluation, we stand it up
as an HTTP service. To launch:

    $ python3 -m uvicorn traq.services.metaod_api:app --host 0.0.0.0

The requirements are in `metaod_requirements_py3.7.txt` and run fine under Python 3.7.17 (a secure,
but non-bugfixed version of Python as of the time of this writing).

## Development

[Poetry](https://python-poetry.org/) is used to manage dependencies (see Usage). It is recommended
that you set up Git hooks using [`pre-commit`](https://pre-commit.com/) to auto-lint your code prior
to committing, which will ensure the CI passes on the server side:

    $ poetry run pre-commit install

This will auto-install a set of Git hooks into the local copy of your repo which will automatically
run each time to you run `git commit`. These hooks include [Black](https://pypi.org/project/black/)
for auto-formatting, [flake8](https://flake8.pycqa.org/en/latest/index.html)
for auto-linting, and [isort](https://pycqa.github.io/isort/) for automatic import sorting.
