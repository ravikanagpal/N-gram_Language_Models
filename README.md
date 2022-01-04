
#  A list of all the resources used


-   https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954 - To find perplexity

-   https://web.archive.org/web/20190805113754/https://web.stanford.edu/~jurafsky/slp3/8.pdf - For deleted interpolation logic

-   https://www.nltk.org - For looking at NLTK APIs

-   https://web.stanford.edu/~jurafsky/slp3/3.pdf - To understand n-gram models

-   https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index - How to sort the list based on tuple value

-   https://www.cs.utexas.edu/~mooney/cs388/slides/equation-sheet.pdf - To find the V (for denominator) in case of laplace smoothing


#  Execution Instructions

## Setup

```sh
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# change directory to the repo where we have requirements file
$ cd N-gram_Language_Models/

# Install python dependencies
$ pip3 install  -r requirements.txt 
```

## Run

Use the following commands in the current directory for various models.

`python3 src/main.py data/train/ data/dev/ output/results_dev_unsmoothed.csv --unsmoothed`

`python3 src/main.py data/train/ data/dev/ output/results_dev_laplace.csv --laplace`

`python3 src/main.py data/train/ data/dev/ output/results_dev_laplace.csv --laplace`


## Data

The assignment's training data can be found in [data/train](data/train) and the development data can be found in [data/dev](data/dev).

## Output

The output is stored in csv files corresponding to each model and can be found in the output directory.

When there is no optional argument passed, the program exits by displaying a message "No optional argument passed. Pass an optional argument to choose a model to run. "

---
