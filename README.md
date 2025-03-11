# Python package 'typeagent'

### Experimental prototype

Working toward a shared understanding of the MVP for structured RAG.

### Sample code

This is an in-progress project aiming at a Pythonic translation of
`TypeAgent/ts/packages/knowPro` and a few related packages to Python.
(Pythonic because it uses Python conventions and types as appropriate.)

- Python class names correspond 1:1 to TS interface or type names.
- Field and method names are converted from camelCase to snake_case.
- I-named interfaces become runtime-checkable Protocol classes;
  other interfaces and structured types become dataclasses.
- Union types remain union types.
- Unions of string literals become Literal types.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project
must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
