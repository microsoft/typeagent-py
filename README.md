# Python package 'typeagent'

### This is an experimental prototype

Working toward a shared understanding of the MVP for structured RAG.

### This is sample code

This is an in-progress project aiming at a Pythonic translation of
`TypeAgent/ts/packages/knowPro` and a few related packages to Python.
(Pythonic because it uses Python conventions and types as appropriate.)

- Python class names correspond 1:1 to TS interface or type names.
- Field and method names are converted from camelCase to python_case.
- I-named interfaces become `Protocol` classes;
  other interfaces and structured types become dataclasses.
- Union types remain union types;
- but unions of string literals become `Literal` types.
- Not all of the TS class hierarchy is translated literally.

### How to build

Tested only on Ubuntu 22 under WSL. Presumably works on most UNIXoids.

- Install Python 3.12 or higher (get it from (python.org)[python.org]
  or run `sudo apt install python3.12`)
- Set your environment to contain the necessary Azure or OpenAI keys.
- Run `make all`
- You should now have a wheel file under `dist/`
- TODO: Upload that wheel to PyPI
- To clean up, run `make clean`


### How to test

- Set your environment to contain the necessary Azure or OpenAI keys.
- Run unit tests: `make test`; this also collects coverage data.
- Review coverage with `coverage report` etc.
- Interactively testing queries: `make demo`
- Comparing to a validated set of questions and expected answers:
  - Obtain or construct a JSON file of q/a pairs and install in testdata
  - Run `make compare` (takes about 5-10 seconds per q/a pair)

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project
must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
