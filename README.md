# UN OCHA COVID-19 International Modeling
## Task 6 - Infectious Diseases
## The Johns Hopkins University Applied Physics Laboratory LLC

# Models

* Cholera: [Modelling cholera transmission dynamics in the presence of limited resources](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6676578/pdf/13104_2019_Article_4504.pdf)
    * This implementation uses system of equations #3.
* Measles: [Controlling measles using supplemental immunization activities: a mathematical model to inform optimal policy](https://www.sciencedirect.com/science/article/pii/S0264410X14016077)
    * This implementation uses a modified version of the system of equations in Section 1 of Appendix A.
* Malaria: [A mathematical model for endemic malaria with variable human and mosquito populations](https://www.sciencedirect.com/science/article/pii/S0895717700001692)
    * This implementation uses system of equations #9.

Each system of differential equations is integrated and solved using [SciPy](https://www.scipy.org/)'s [solve_ivp()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) function.

# Installation

Python 3.6+ is required. To install dependencies:

`pip3 install -r requirements.txt`

# Usage

### Basic Usage

`$ python3 Cholera.py CholeraScenario1.xlsx`

`<model>.py` is the filepath of the model you want to run.

`CholeraScenario1.xlsx` is the path to the input Excel file. This argument is required.

### Optional Command Line Arguments
`-o`, `--output_dir` is the path to the directory where results will be stored. The default is the directory where `<model>.py` is located.

`--dash` is an optional flag that will start an interative Dash chart in a web browser.

`--time` is the number of time steps for the model simulation to run in Dash. The current Dash implementation will only use parameter values from the first column of the time series in the Excel file.

`--port` is the port number for Dash to use.

`-i`, `--image` will save static screenshots of the plots to `--output_dir`.

# Input

The input to a model is a templated Excel file that contains values for the model's parameters and initial conditions.

* There should be 1 Excel file for each model-scenario combination. E.g.,
    * CholeraScenario1.xlsx
    * MeaslesScenarioBest.xlsx
    * MeaslesScenarioWorst.xlsx
    * MalariaScenarioFlood.xlxs
* Within each file, there should be 1 sheet (tab) for each geographic region of interest, which will have its own set of parameter values.
    * As many sheets can be added to a file as is desired.
    * The model simulation will be run for each sheet.
* Within each sheet, there will be a table for its parameters and initial conditions.
    * Rows represent configurable parameters and initial conditions.
    * Columns represent different  intervals or "shocks" within the piecewise-continuous time series.
        * As many columns can be added to a table as is desired.
        * The model's system of equations will be integrated across the interval specified by each column.
        * The heading for each column is the time step `b` that the interval *ends* at.
        * `b` must be a positive integer greater than the previous column's heading.
        * The interval of integration for that column will be `(a, b - 1)` where a is the previous column's heading or 0 if there is no previous column.
        * The initial conditions for the first column must be specified.
    * Initial conditions for each sheet must be specified in the first column only.
        * Values for the initial conditions in any subsequent columns will be ignored, and the end state conditions from the end of the *previous* time interval will be used instead as the new "initial" conditions.
