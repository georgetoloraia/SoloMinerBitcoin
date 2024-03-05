import aiohttp
import asyncio

async def fetch_block_data(session, semaphore, block_url, height):
    async with semaphore:
        async with session.get(block_url) as response:
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', '10'))  # Default to 10 seconds if Retry-After header is missing
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                return await fetch_block_data(session, semaphore, block_url, height)

            if response.status != 200:
                print(f"Error: Failed to fetch block data for height {height}. Status code: {response.status}")
                return

            try:
                block_data = await response.json()
                nonce = block_data['blocks'][0]['nonce']
                if nonce is not None:
                    print(f"Block Height: {height}, Nonce: {nonce}")
                    with open('nonce.txt', 'a') as f:
                        f.write(f"{nonce}\n")
                else:
                    print(f"Block Height: {height}, Nonce not found")
            except ValueError:
                print(f"Error: Response from {block_url} does not contain valid JSON")
                print(f"Response content: {await response.text()}")

async def extract_nonces():
    try:
        start_height = 0
        end_height = 832290
        esplora_api = 'https://blockchain.info'
        concurrency = 100  # Adjust the concurrency level as needed
        batch_size = 10  # Adjust the batch size as needed

        semaphore = asyncio.Semaphore(concurrency)

        async with aiohttp.ClientSession() as session:
            for height in range(start_height, end_height + 1, batch_size):
                tasks = []
                for h in range(height, min(height + batch_size, end_height + 1)):
                    block_url = f'{esplora_api}/block-height/{h}'
                    task = asyncio.ensure_future(fetch_block_data(session, semaphore, block_url, h))
                    tasks.append(task)

                await asyncio.gather(*tasks)

    except aiohttp.ClientError as e:
        print(f"Error: {e}")

# Example usage
asyncio.run(extract_nonces())
