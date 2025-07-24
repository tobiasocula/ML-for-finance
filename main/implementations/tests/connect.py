import ib_async as iba
import asyncio

async def main():
    ib = iba.IB()

    # Connect to IB API Gateway or TWS
    await ib.connectAsync(host="127.0.0.1", port=4002)

    # Optionally wait a second to allow data initialization
    await asyncio.sleep(1)

    # Get list of managed accounts
    accounts = ib.managedAccounts()
    if not accounts:
        print("No managed accounts found.")
        ib.disconnect()
        return

    account = accounts[0]
    print(f"Using account: {account}")

    # Request account summary for this account
    summary = await ib.accountSummaryAsync(account)
    for item in summary:
        print(f"{item.tag}: {item.value}")

    ib.disconnect()

asyncio.run(main())
