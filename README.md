# OPTrading

Source code for "OPTrading".

[Online Demonstration](https://quant-nlp.github.io/OPTrading/) is available.

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

# e.g. bash scripts/run.sh GOOG T
```

### Example Output

```
crawling GOOG tweets ...
crawling T tweets ...

Historical Testing Accuracy :  64.71 %
Historical return on investment :  12.91 %
Historical investment risk :  0.92 %

Decision :
Long T Short GOOG
```
