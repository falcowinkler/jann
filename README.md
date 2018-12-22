# just another neural network

A tiny neural network written in clojure that learns to recognize mnist digits,
without any libraries except `clojure.core.matrix`.
Purely for fun and learning.

## Usage

Run it with

```bash
lein run
```

You should see the accuracy rising

```bash
Training epoch  0
1427/2000 correct classifications
Training epoch  1
1675/2000 correct classifications
...
Training epoch  10
1728/2000 correct classifications
```

Test it with

```bash
lein test
```

## Data

Data is expected as two dimensional clojure vector.
Because i was lazy i have just dumped the 10000 images
from [here](https://github.com/keorn/clj-mnist), that is
why the network currently just reaches 90% accuracy.

## TODO's

- load mnist from binary files
- improve accuracy with advanced techniques
- add possibility to classify an image file with the learned model