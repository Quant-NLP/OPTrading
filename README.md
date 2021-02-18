# OPTrading

Source code for "OPTrading".

## Requirements

`python >= 3.6.0`, Install all the requirements with pip.

```
$ pip install -r requirements.txt
```

We require [pystocktwits](https://github.com/khmurakami/pystocktwits) for crawling data from [Stocktwits](https://stocktwits.com/).


## Getting Started

```
bash scripts/run.sh stock1 stock2

# e.g. bash scripts/run.sh AAPL GOOG
```

### Example Output

```
crawling AAPL tweets ...
crawling GOOG tweets ...

Historical Testing Accuracy :  60.78431372549019 %
Historical return on investment :  107.89564439891353 %
Historical investment risk :  1.4847406578104394 %

Decision :
Long GOOG Short AAPL
```
