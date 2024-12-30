import asyncio
import time

async def say_after(delay, what):
    await asyncio.sleep(delay)
    return what

async def main():
    task1 = asyncio.create_task(
        say_after(3, 'hello'))

    task2 = asyncio.create_task(
        say_after(4, 'world'))

    print(f"started at {time.strftime('%X')}")

    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    print(await task1)
    print(await task2)

    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())