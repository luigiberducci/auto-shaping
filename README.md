# Auto-Shaping
Specification-based library for automatic reward shaping.

## Methods
| Method                               | Dense Signal       | Multi-Objective     | Objective Prioritization | Implemented        |
|--------------------------------------|--------------------|---------------------|--------------------------|--------------------|
| TLTL<sup>[1]</sup>                   | :x:                | :x:                 | :x:                      | :heavy_check_mark: |
| BHNR<sup>[2]</sup>                   | :heavy_check_mark: | :x:                 | :x:                      | :x:                |
| MORL<sup>[3]</sup>                   | :grey_question:    | :heavy_check_mark:  | :grey_question:          | :x:                |
| HPRS<sup>[4]</sup>                   | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark:       | :x:                |
| PAM<sup>[4]</sup>                    | :x:                | :heavy_check_mark:  | :heavy_check_mark:       | :x:                |
| Rank-Preserving Reward<sup>[5]</sup> | :x:                | :heavy_check_mark:  | :heavy_check_mark:       | :x:                |

:heavy_check_mark: Yes

:grey_question: Possible

:x: Not supported

## Examples
 - [ ] Integration with gymnasium
 - [ ] Integration with stable-baselines3
 - [ ] Integration with safety-gym
 - [ ] Use of predefined env specs
 - [ ] Use of custom env specs

# References
[1] "Reinforcement learning with temporal logic rewards." Li, et al. IROS 2017.

[2] "Structured reward shaping using signal temporal logic specifications." Balakrishnan, et al. IROS 2019.

[3] "Multi-objectivization of reinforcement learning problems by reward shaping." Brys, et al. IJCNN 2014.

[4] "Hierarchical Potential-based Reward Shaping." Berducci, et al. Under Review.

[5] "Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles." Veer, et al. ICRA 2023.
