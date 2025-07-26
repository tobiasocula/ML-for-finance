import asyncio
from ib_async import IB, Stock

async def main():
    ib = IB()
    # Connect asynchronously to IB Gateway or TWS (adjust port if needed)
    await ib.connectAsync(host="127.0.0.1", port=4002)

    ib.reqMarketDataType(3)
    
    # Define a contract for AAPL stock on SMART exchange priced in USD
    contract = Stock(symbol='AAPL', exchange='SMART', currency='USD')
    
    qualified_contracts = await ib.reqContractDetailsAsync(contract)

    if not qualified_contracts:
        print("Contract qualification failed. No details returned.")
        ib.disconnect()
        return
    
    # Use the first qualified contract returned
    qualified_contract = qualified_contracts[0].contract
    print(f"Qualified contract conId: {qualified_contract.conId}")

    # Request live market data subscription
    ticker = ib.reqMktData(qualified_contract, '', False, False)
    
    # Stream live quotes for 30 seconds, printing every second
    for _ in range(100):
        await asyncio.sleep(1)
        # Check if data has been received and print it
        if ticker.last:
            print(f"AAPL: Last=${ticker.last}, Bid=${ticker.bid}, Ask=${ticker.ask}")
        else:
            print("Waiting for market data...")
    
    # Disconnect cleanly
    ib.disconnect()

asyncio.run(main())
