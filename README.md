platform: Mac OS X

python version: Python 3.7.2

```bash
usage: learnr1.py [-h] [--learner {naive-bayesian}] [--estimator ESTIMATOR]
                  [--dataset {itg,orig,resplit-itg,resplit}]

heart anomaly detector

optional arguments:
  -h, --help            show this help message and exit
  --learner {naive-bayesian}, -l {naive-bayesian}
                        what learner would you like to use?
  --estimator ESTIMATOR, -e ESTIMATOR
                        m-estimator
  --dataset {itg,orig,resplit-itg,resplit}, -d {itg,orig,resplit-itg,resplit}
                        choose a dataset to learn and test
```