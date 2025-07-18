# Reproducibility-Evaluating-Plan-based-Adaptive-Query-Processing

**Paper:** Evaluating Plan-based Adaptive Query Processing: \[Experiments & Analysis\] [[link](tbd)]

#### Source Code of Systems
- Repository: AQP-DuckDB (https://anonymous.4open.science/r/duckdb-5319/README.md)
- Repository: AQP-PostgreSQL (https://anonymous.4open.science/r/PostgreSQL-12\_3-8A0F/README)

#### Source code of Benchmarks and Datasets
- Repository: JOB benchmark (https://anonymous.4open.science/r/join-order-benchmark-BA30/README.md)
- Repository: DSB benchmark (https://anonymous.4open.science/r/dsb-for-AQP-6DB3/README.md)

#### Hardware Info to Reproduce Experiment Results

- Processor: Intel(R) Xeon(R) E-2236 CPU @ 3.40GHz
- Memory: 64 GB DDR4 RAM @ 2666 MHz
- Disk: 1 × 931.5 GB SATA HDD (6GB/s)

#### Experimentation Info

- Environment: Ubuntu 22.04.5 LTS is used for the experiments.
- Compiler: gcc-11.4.0/g++-11.4.0
- Measurement tool: hyperfine (https://github.com/sharkdp/hyperfine)

Please note that we assume `python3`, `python3-venv`, and `anaconda` to be pre-installed. We advise to run the experiments on a clean Ubuntu 22.04 (virtual) machine.
