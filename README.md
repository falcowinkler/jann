# just another neural network

A tiny neural network written in clojure that learns to recognize mnist digits.
Purely for fun and learning.

## Usage

Run it with

```bash
lein run
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