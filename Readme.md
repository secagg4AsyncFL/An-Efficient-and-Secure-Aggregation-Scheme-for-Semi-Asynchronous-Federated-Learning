# An Efficient and Secure Aggregation Scheme for Semi-Asynchronous Federated Learning

Our scheme is implemented in flgo/algorithm/secfedbuff.py, and secfedbuff_GIA_test.py can evaluate the performance of our scheme under the GIA attack.

## Requirements


## Running experiments

We provide the essential dependencies in requirements.txt, which can help you build up conveniently.

`pip install -r requirements.txt`

And the completely environment exported by Anaconda is provided in environment.yaml, which can help you build up conveniently.

`conda env create -f environment.yml`


## Start the experiment

Due to its complexity, the sample startup code is provided in the test.py file, including fundamental configurations for training, dataset and model selection, data distribution, and algorithms.

## Notice

To facilitate measurement of experimental data, we have implemented a demo for the paper's experiments under the `./Experiment Demo` directory, which can be run directly.

All code runs by default in a standalone environment without SGX. To run in an SGX environment, you must install the SGX SDK and Graphene (now renamed Gramine) on an SGX-enabled CPU.


