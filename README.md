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

Please note that we assume `python3`, `python3-venv`, and `anaconda` to be pre-installed. We advise running the experiments on a clean Ubuntu 22.04 (virtual) machine.


## Usage
The experiments include PostgreSQL and DuckDB running on the JOB and DSB benchmarks. For PostgreSQL, it loads data from CSV files immediately. For DuckDB, we export the dataset from PostgreSQL to a `.db` file and reload it for query execution. The end-to-end time includes the loading time (included in the "Other" part of the performance breakdown).

### Environment Variables Across the Reproduction

Here are a few environment configurations. I write them in `~/.bashrc`.
```bash
export PREFIX=YOUR_BIN_PATH

Project_path=$PREFIX
export PATH=$PATH:$Project_path/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$Project_path/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$Project_path/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$Project_path/include

###### PostgreSQL Alias
alias pg_start='pg_ctl start -l $Project_path/logfile -D $Project_path/data'
alias pg_stop='pg_ctl stop -D $Project_path/data -m smart -s'
alias pg_log='vi $Project_path/logfile'
alias rm_pg_log='rm $Project_path/logfile'

###### DuckDB configurations
export PATH=$PATH:YOUR_DUCKDB_BIN_PATH
export PATH="/usr/lib/ccache:$PATH" # ccahe is used for speeding up compilation

###### JOB_PATH
export JOB_PATH=YOUR_JOB_PATH

###### DSB_PATH
export DSB_PATH=YOUR_DSB_PATH
```

### Install PostgreSQL
We need first to install vanilla PostgreSQL 12.3 (https://anonymous.4open.science/r/PostgreSQL-12\_3-8A0F/README)

#### Compilation
```bash
export PREFIX=YOUR_BIN_PATH
git clone xxx && cd PostgreSQL-12.3/
mkdir build && cd build
sudo apt install libreadline-dev zlib1g-dev make bison flex gawk

../configure --prefix=$PREFIX
make -j32 && make check
sudo make install
```

#### Setup
```bash
# create database
pg_ctl -D $PREFIX/data initdb # initdb -D $PREFIX/data

# start server
pg_ctl start -l $PREFIX/logfile -D $PREFIX/data

# stop server
pg_ctl stop -D $PREFIX/data -m smart -s
```

### Install DuckDB
We also need to install vanilla DuckDB v0.10.1 (https://anonymous.4open.science/r/duckdb-5319/README.md)

#### Compilation
```bash
git clone xxx && cd duckdb
mkdir build && cd build
cd ../ && make clean && GEN=ninja VERBOSE=1 make 2>&1|tee -a compile.log
```

### Prepare JOB benchmark
Please also refer to the JOB benchmark (https://anonymous.4open.science/r/join-order-benchmark-BA30/README.md)

#### PostgreSQL
```bash
git clone xxx && cd JOB4AQP
wget http://event.cwi.nl/da/job/imdb.tgz
tar -zxvf imdb.tgz
mkdir csv && mv *.csv csv/.

pg_start
createuser imdb
createdb imdb
psql -U imdb -d imdb -f schema.sql
psql -U imdb -d imdb -f import_csv.sql # please replace the `CURRENT_PATH` with the correct path!!!
psql -U imdb -d imdb -f fkeys.sql
psql -U imdb -d imdb -f fkindexes.sql

###### verification
bash ./execute_queries.sh

pg_stop
```

#### DuckDB
```bash
# go to duckdb dir
git checkout query_split && cd measure
# note to use vanilla DuckDB to generate .db files.
bash ./prepare_job.sh

###### verification
bash ./run_duckdb_job.sh
```


### Prepare DSB benchmark
Please also refer to the DSB benchmark (https://anonymous.4open.science/r/dsb-for-AQP-6DB3/README.md)

#### PostgreSQL
```bash
git clone xxx && cd DSB4AQP

cd code/tools/
make clean && make # sudo apt install gcc-9

# prepare python environment
conda create -n dsb python=3.10
conda activate dsb
pip3 install -r ../../scripts/requirements.txt

python ../../scripts/generate_dsb_db_files.py 10 # data files are in code/tools/out_10
# OR python ../../scripts/generate_dsb_db_files.py 100 # data files are in code/tools/out_100
python ../../scripts/generate_workload.py postgres # queries are in code/tools/1_instance_out
# OR python ../../scripts/generate_workload.py duckdb # queries are in code/tools/1_instance_out

# prepare and run pg
pg_start # start postgres server
createdb dsb_10
# OR createdb dsb_100

cd code/tools/

python ../../scripts/load_data_pg.py 10
# OR python ../../scripts/load_data_pg.py 100
psql -U postgres -d dsb_10 -f tpcds_ri.sql
# OR psql -U postgres -d dsb_100 -f tpcds_ri.sql

# first modify the `bin_path =` in `../../scripts/create_index_pg.py`, set to your postgres bin path
python ../../scripts/create_index_pg.py 10
# OR python ../../scripts/create_index_pg.py 100

cd ../../scripts

bash ./prepare_QuerySplit_queries.sh

bash ./execute_dsb_pg.sh Official 10
# OR bash ./execute_dsb_pg.sh Official 100
bash ./execute_dsb_pg.sh QuerySplit 10
# OR bash ./execute_dsb_pg.sh QuerySplit 100

bash ./export_csv_pg.sh 10
# OR bash ./export_csv_pg.sh 100

pg_stop
```

#### DuckDB
```bash
# stay in DSB4AQP/scripts; note to use vanilla DuckDB to generate .db files.
bash ./prepare_duckdb.sh 10
cp dsb_10.db duckdb_measure_dir
# OR bash ./prepare_duckdb.sh 100
# OR cp dsb_100.db duckdb_measure_dir
```

-----------------------
Now, you have settled all the datasets and benchmarks! Congratulations!!!
-----------------------


### Experiments in PostgreSQL
Please also refer to the AQP-PostgreSQL (https://anonymous.4open.science/r/PostgreSQL-12\_3-8A0F/README)
```bash
# go to PostgreSQL dir
git checkout subquery_scan

# measure JOB
sudo rm -rf job_result/
bash ./measure_job.sh && bash ./measure_breakdown_time_job.sh

# measure DSB
sudo rm -rf dsb_10_result/
bash ./measure_dsb.sh 10 && bash ./measure_breakdown_time_dsb.sh 10

sudo rm -rf dsb_100_result/
bash ./measure_dsb.sh 100 && bash ./measure_breakdown_time_dsb.sh 100
```

### Experiments in DuckDB
Please also refer to the AQP-DuckDB (https://anonymous.4open.science/r/duckdb-5319/README.md)
```bash
# go to DuckDB dir
# confirm now you are on the query_split branch
cd measure
# measure JOB
sudo rm -rf job_result/
bash ./measure_job.sh && bash ./measure_breakdown_time_job.sh

# measure DSB
sudo rm -rf dsb_result_10/
bash ./measure_dsb.sh 10 && bash ./measure_breakdown_time_dsb.sh 10

sudo rm -rf dsb_result_100/
bash ./measure_dsb.sh 100 && bash ./measure_breakdown_time_dsb.sh 100
```


-----------------------
Now, you have finished all the experiments and got all the numbers! Congratulations!!!
-----------------------

Please feel free to contact us by email or report any issues if you encounter a problem.


## Figure plotting
```bash
# prepare python environment
cd scripts
conda create -n eval_aqp python=3.10
conda activate eval_aqp
pip3 install -r requirements.txt
python3 experimental_figures.py # it's okay to have some warnings about `brokenaxes`
```

### Additional Figures (as listed in the appendix)
We provide scripts for additional figures, including deviation for each system/setting, performance breakdowns, and query-by-query time comparisons in a bar chart, among others.
```bash
python3 plot_duckdb_end2end_results.py JOB && python3 plot_duckdb_end2end_results.py DSB_10 && python3 plot_duckdb_end2end_results.py DSB_100
```

## Raw Data (on two different servers)
We measured all of the experiments on two different servers. 

One is the server mentioned in the paper, featuring an Intel(R) Xeon(R) E-2236 CPU @ 3.40 GHz, 64 GB of DDR4 RAM at 2666 MHz, and 1 × 931.5 GB SATA HDD (6 GB/s).

The other is a server with an AMD EPYC 7501 @ 2.0 GHz, 256 GB (16 * 16 GB) DDR4 @ 2666 MHz, and 1 x 931.5G Samsung NVME (SSD with 6 GB/s).

We will upload our raw data for your convenience once the paper is accepted.

## Citation

If you use our experimental evaluation, please cite our paper.

TBD.
