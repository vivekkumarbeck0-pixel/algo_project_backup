from readers.price_reader import PriceReader


reader = PriceReader()

print("=" * 50)
print("PRICE READER OUTPUT")
print("=" * 50)

result = reader.read("data/tradingview.png")

for text in result:
    print(text)