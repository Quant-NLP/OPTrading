# OPTrading

Source code for "OPTrading".

## Requirements

`python >= 3.6.0`, Install all the requirements with pip.

```
$ pip install -r requirements.txt
```

We require [pystocktwits](https://github.com/khmurakami/pystocktwits) for crawling data from [Stocktwits](https://stocktwits.com/).

In our experiment, we use the partial [StockNet](https://github.com/yumoxu/stocknet-dataset) data, which is collected from Twitter.

## Getting Started

```
bash scripts/run.sh stock1 stock2

# e.g. bash scripts/run.sh AAPL GOOG
```

### Example Output

```
crawling AAPL tweets ...
crawling GOOG tweets ...

Historical Testing Accuracy :  60.78 %
Historical return on investment :  107.9 %
Historical investment risk :  1.48 %

Decision :
Long GOOG Short AAPL
```
