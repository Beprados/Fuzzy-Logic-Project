# Fuzzy-Logic-Project
Codes for computing design in Fuzzy Logic. Implements the Fuzzy Adaptive Differential Evolution (FaDE) algorithm, first presented in the paper by Liu and Lampinen. In particular, the technique introduces a Fuzzy Logic System for self-regulation of optimization parameters at runtime. Some test functions are defined in order to define a minimum set of parameters for performance comparison with the original version of Differential Evolution.

Fuzzy Logic Systems emerge as an alternative for computational intelligence to infer, based upon Mamdani Method, the parameters that govern the mutation rate (F) and genetic crossover (CR). Once computational memory (even though restricted only to a single previous generation) and logic systems are granted, one may expect clear improvement in reproduction strategies by succecive populations, especially at exploration and exploitation - preventing premature stagnation at local minima. The following animation shows the evolution of the same initial population through two distintict methods: the classic Differential Evolution, and it's Fuzzy variant applied to the Griewank test function - with global minima at (0,0) point.

![me](https://github.com/Beprados/Fuzzy-Logic-Project/blob/main/griewank_de_fade.gif)

Besides, the F and CR changing behaviour at run time are plotted along the FaDE minimization of a "needle in a haystack"-type test function, the Easom function.

![me](https://github.com/Beprados/Fuzzy-Logic-Project/blob/main/easom_fade_with_fcr.gif)
