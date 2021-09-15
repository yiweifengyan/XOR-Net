# The XOR-Net Paper Experiment Source Code

### Target Platform:
 - Hardware: [GreenWaves GAP8](https://greenwaves-technologies.com/gap8_mcu_ai/).
 - Compiling Environment: GAP_SDK with PULP OS suport.

### Contents:
- main.c: The experiment source code.
- Makefile: The Makefile that configs GAP8. 
- Example Result.txt: The command line output of one example execution.

### Usage:
- Install [GreenWaves GAP_SDK](https://github.com/GreenWaves-Technologies/gap_sdk).
- Config the Makefile to include the correct "pulp_rules.mk".
- Run this command under the source code folder: "make clean all run".

### How to execute it on Windows/Linux/Mac
- Extract the convolution functions.
- Replace the B_int_r (insert a single bit) and B_popc (popcnt) with platform specific APIs.
- Write your own benchmark code and execute.

### References
 - https://github.com/GreenWaves-Technologies/gap_sdk
 - https://github.com/GreenWaves-Technologies/benchmarks


 If you find this repository useful, please cite:  
 
 ```
 @inproceedings{zhu2020xor,  
  title={XOR-Net: An Efficient Computation Pipeline for Binary Neural Network Inference on Edge Devices},  
  author={Zhu, Shien and Duong, Luan HK and Liu, Weichen},
  booktitle={2020 IEEE 26th International Conference on Parallel and Distributed Systems (ICPADS)},
  pages={124--131},
  year={2020},
  organization={IEEE}
}
 ```





