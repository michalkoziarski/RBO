# Radial-Based Oversampling

Python code associated with the paper Radial-Based Oversampling for Noisy Imbalanced Data Classification.

# Usage

Tested on Python 2.7.9.

Because of a high computational cost of the RBO method, code was written to take advantage of a parallel processing, running specific trials of the experiment simultaneously. Because of that, some additonal steps are requried to run the experiment.

To download all of the necessary datasets:

```python datasets.py```

To initialize the databases:

```python databases.py```

To schedule the experiments associated with the [preliminary|final] analysis:

```python experiments/schedule_[preliminary|final].py```

To start a runner, pulling unfinished trials until there are none left (note that several runners can operate simultaneously):

```python run.py```

To export the results from a previously initialized database into a CSV file:

```python databases.py```
