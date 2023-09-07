[![Logo](docs/logo.svg)](https://github.com/luigiberducci/auto-shaping)

<h2 align="center">
Specification-based library for automatic reward shaping.
</h2>

<p align="center">
<a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://python.org"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


## Methods
| Method                               | Dense Signal       | Multi-Objective     | Objective Prioritization | Implemented         |
|--------------------------------------|--------------------|---------------------|--------------------------|---------------------|
| TLTL<sup>[1]</sup>                   | :x:                | :x:                 | :x:                      | :heavy_check_mark:  |
| BHNR<sup>[2]</sup>                   | :heavy_check_mark: | :x:                 | :x:                      | :x:                 |
| HPRS<sup>[4]</sup>                   | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark:       | :heavy_check_mark:  |
| PAM<sup>[4]</sup>                    | :x:                | :heavy_check_mark:  | :heavy_check_mark:       | :x:                 |
| Rank-Preserving Reward<sup>[5]</sup> | :x:                | :heavy_check_mark:  | :heavy_check_mark:       | :x:                 |

:heavy_check_mark: Yes

:grey_question: Possible

:x: Not supported

## Specification Language
The task specification consists of a set of requirements, as in [4]. The requirement syntax is as follows:
```
formula ::= f(state) ~ 0
requirement ::= ensure <formula> | achieve <formula> | conquer <formula> | encourage <formula>
```

where `f` is a function of the state dictionary `state` 
and `~` is a comparison operator in `<`, `<=`, `>`, `>=`.

# Citation
If you use this code in your research, please cite the following paper:
```
@misc{berducci2022hierarchical,
    title={Hierarchical Potential-based Reward Shaping from Task Specifications}, 
    author={Luigi Berducci and Edgar A. Aguilar and Dejan Ničković and Radu Grosu},
    year={2022},
    eprint={2110.02792},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

# References
[1] "Reinforcement learning with temporal logic rewards." Li, et al. IROS 2017.

[2] "Structured reward shaping using signal temporal logic specifications." Balakrishnan, et al. IROS 2019.

[3] "Multi-objectivization of reinforcement learning problems by reward shaping." Brys, et al. IJCNN 2014.

[4] "Hierarchical Potential-based Reward Shaping." Berducci, et al. Under Review.

[5] "Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles." Veer, et al. ICRA 2023.
