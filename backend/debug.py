import httpx
import asyncio
import json

async def main():
    async with httpx.AsyncClient() as client:
        async with client.stream('POST', 'http://localhost:8000/chat', json={'query': 'What is the secret code for the nuclear launch?'}) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    try:
                        data_str = line[6:].strip()
                        if data_str:
                            parsed = json.loads(data_str)
                            print(parsed)
                    except Exception as e:
                        print("Error parsing:", e)

asyncio.run(main())
