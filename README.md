# Distributed SGD Matrix Factorization (Spark)

## How to Run

`spark-submit SPARK-PARAMS dsgd_mf.py FACTORS WORKERS ITERATIONS BETA LAMBDA INPUT_PATH W_PATH H_PATH`
Each line in the input file is assumed to be `i,j,value`.

Here is an example:

`spark-submit --master local[8] --executor-memory 5G --driver-memory 5G dsgd_mf.py 16 8 100 0.5 1 input.csv w.csv h.csv`
