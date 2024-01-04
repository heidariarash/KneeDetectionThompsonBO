# Knee Detection in Bayesian Multi-Objective Optimization Using Thompson Sampling

Welcome to the GitHub repository for our paper, "Knee Detection in Bayesian Multi-Objective Optimization Using Thompson Sampling." The corresponding paper can be accessed here.

## Project Structure
### Source Code
+ acqf.py: Implementation of the acquisition functions.
+ problems.py: Definitions for different optimization problems.
+ solver.py: Aggregated definitions in a function called solver to streamline execution.
+ main.py: Execution script to run the optimization code.

### Configuration
To customize the optimization runs, modify the config.py file. Key settings include:

+ n_var: Specifies the input dimensionality.
+ knees: Defines the number of knees in problems where this is a parameter (DEB2DK, DO2DK, CKP).
+ s: Specific to the DO2DK problem, specifying the s parameter.
+ problem: Select the desired problem to be optimized from options: DEB2DK, DO2DK, DEB3DK, CKP, CBEAM, CYCLONE, DTLZ7, ZDT3.
+ EA_util: Chooses the auxiliary evolutionary algorithm in HVKTS-EA from options: NSGAII and SMSEMOA.
+ method: Specifies the acquisition function to be used for optimization from options: HVKTS, HVKTS-EA, EHVI, HV-KNEE.
+ repetitions: Specifies the number of optimization repetitions using the specified settings. Extract statistical measures at the end, such as median and quantiles for multiple repetitions.

## Getting Started
1. Clone the repository: `git clone https://github.com/heidariarash/KneeDetectionThompsonBO.git`
2. Navigate to the project directory: `cd [repository_directory]`
3. Modify the config.py file to tailor the optimization parameters.
4. Run the optimization script: python main.py

Feel free to explore and adapt the code for your specific needs. If you have any questions or encounter issues, please let us know by creating an issue in this repository.

Happy optimizing!