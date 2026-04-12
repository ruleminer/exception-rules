## Experiment 1

An experiment focused on knowledge discovery and testing the generation of exception rules. As part of this experiment, exception rule generation algorithms will be run on full datasets. As a result, two files will be produced: `exceptions_summary.csv` and `exceptions_details.csv`.  
The first file will contain a summary: `dataset`, `algorithm`, `number of rules`, `number of exceptions` (and for classification: number of type 1 exceptions, number of type 2 exceptions, number of AR exceptions, number of None exceptions).  
The second file will contain details: `dataset`, `algorithm`, `id`, `CR`, `RR`, `ER`, `MY_MEASURE` (and for classification: `type`, `GACE`, `RI`).

Experiments for different problem types will be marked as:

* `clf` - classification
* `reg` - regression
* `srv` - survival

The experiments will be conducted for:

* `clf` - algorithm3
* `reg` - algorithm3 and algorithm 4 (a modification of algorithm 3 where an exception occurs when the exception conclusion is outside the confidence interval of CR and RR)
* `srv` - algorithm3 and algorithm 4 (a modification of algorithm 3 with modified search for survival curves that should be above the KM of CR and RR when they are below the KM of the whole dataset or below when they are above)




