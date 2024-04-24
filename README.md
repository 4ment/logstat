# logstat

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`logstat` is a simple program replicating some of [tracer]'s functionalities. It takes a posterior sample as input and outputs the effective sample size (ESS) and statistics such as the mean, meadian, and 95% higher posterior density (HPD).

### Dependencies
 - [numpy]
 - [pandas]

 ### Installation
To install logstat from source you can run
```bash
pip install git+https://github.com/4ment/logstat
```

or

```bash
git clone https://github.com/4ment/logstat
pip install logstat/
```

Check install
```bash
logstat --help
```

## Quick start
`logstat` parses log files generated by program such as `beast`.

```bash
logstat samples.log --burnin 0.2
```

Multiple sample files can be provided and the burnin can be specified for each of them using the same order.

```bash
logstat samples1.log samples2.log --burnin 0.2 --burnin 0.3
```


## License

Distributed under the GPLv3 License. See [LICENSE](LICENSE) for more information.

## Acknowledgements

`logstat` makes use of the following libraries and tools, which are under their own respective licenses:

 - [numpy]
 - [pandas]

[numpy]: https://github.com/numpy/numpy
[pandas]: https://github.com/pandas-dev/pandas
[tracer]: https://github.com/beast-dev/tracer