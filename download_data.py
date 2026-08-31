"""
Fetches the public datasets used across this repository into ./data/.

Run once before executing any notebooks:
    python download_data.py

Reuses the same real, public data sources validated in the companion
quant-portfolio repo.
"""
import os
import urllib.request

SOURCES = {
    "data/nyse_prices.csv":
        "https://raw.githubusercontent.com/kyi3081/stock-analysis/master/prices-split-adjusted.csv",
    "data/btc_price.csv":
        "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv",
    "data/eth_price.csv":
        "https://raw.githubusercontent.com/blockchain-unica/ethereum-ponzi/master/price-eth-usd.csv",
    "data/fx_daily.csv":
        "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv",
    "data/bitcoin.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/bitcoin.csv",
    "data/litecoin.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/litecoin.csv",
    "data/ethereum.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/ethereum.csv",
    "data/xrp.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/xrp.csv",
    "data/dogecoin.csv":
        "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/dogecoin.csv",
}

def main():
    os.makedirs("data", exist_ok=True)
    for dest, url in SOURCES.items():
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("Done. You can now run the notebooks in each project folder.")

if __name__ == "__main__":
    main()
