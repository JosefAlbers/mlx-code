import asyncio
import httpx
import time
import sys
import argparse
SERVER = 'http://localhost:8000'
COLORS = ['\x1b[92m', '\x1b[94m', '\x1b[93m', '\x1b[95m', '\x1b[96m']
RESET = '\x1b[0m'
BOLD = '\x1b[1m'
DIM = '\x1b[2m'
PROMPTS = [('A', 'Explain how neural networks learn in simple terms:'), ('B', 'Write a short poem about the ocean:'), ('C', 'def quicksort(arr):'), ('D', 'The history of the Roman Empire began'), ('E', 'What are three benefits of exercise?')]

async def stream_request(client, label, prompt, max_tokens, fancy, color, t0):
    sent_at = time.perf_counter() - t0
    prefix = f'{color}[{label}]{RESET}' if fancy else f'[{label}]'
    print(f'\n{BOLD}{prefix} +{sent_at:.2f}s | {prompt[:60]}{RESET}')
    print(f'{prefix} >> ', end='', flush=True)
    first_token_at = None
    count = 0
    try:
        async with client.stream('POST', f'{SERVER}/generate', json={'prompt': prompt, 'max_tokens': max_tokens, 'stream': True}, timeout=120) as resp:
            async for chunk in resp.aiter_text():
                if first_token_at is None:
                    first_token_at = time.perf_counter() - t0
                c = f'{color}{chunk}{RESET}' if fancy else chunk
                print(c, end='', flush=True)
                count += 1
    except Exception as e:
        print(f'\n{prefix} ERROR: {e}', flush=True)
        return
    done_at = time.perf_counter() - t0
    ttft = f'{first_token_at - sent_at:.2f}s' if first_token_at else 'n/a'
    tail = f'{DIM}{color}' if fancy else ''
    print(f'\n{tail}{prefix} done +{done_at:.2f}s | {count} chunks | ttft {ttft}{RESET}')

async def run_overlapping(fancy, max_tokens):
    print('\n' + '=' * 60)
    print('OVERLAP — all 5 fire at once, batched in one forward pass')
    print('=' * 60)
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[stream_request(client, label, prompt, max_tokens, fancy, COLORS[i % len(COLORS)], t0) for i, (label, prompt) in enumerate(PROMPTS)])
    print(f'\nAll done in {time.perf_counter() - t0:.2f}s')

async def run_staggered(fancy, max_tokens):
    print('\n' + '=' * 60)
    print('STAGGER — requests arrive at different times, slot into batch')
    print('=' * 60)
    delays = [0.0, 2.0, 0.5, 4.0, 1.5]
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:

        async def delayed(i, label, prompt, delay):
            await asyncio.sleep(delay)
            await stream_request(client, label, prompt, max_tokens, fancy, COLORS[i % len(COLORS)], t0)
        await asyncio.gather(*[delayed(i, label, prompt, delays[i]) for i, (label, prompt) in enumerate(PROMPTS)])
    print(f'\nAll done in {time.perf_counter() - t0:.2f}s')

async def run_sequential(fancy, max_tokens):
    print('\n' + '=' * 60)
    print('SEQUENTIAL — one at a time, no batching (baseline)')
    print('=' * 60)
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        for i, (label, prompt) in enumerate(PROMPTS):
            await stream_request(client, label, prompt, max_tokens, fancy, COLORS[i % len(COLORS)], t0)
    print(f'\nAll done in {time.perf_counter() - t0:.2f}s')

async def main(mode, fancy, max_tokens):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f'{SERVER}/health', timeout=5)
            info = r.json()
            print(f'Server OK | active sequences: {info.get('active_sequences', '?')}')
    except Exception as e:
        print(f'Cannot reach server: {e}')
        sys.exit(1)
    if mode == 'overlap':
        await run_overlapping(fancy, max_tokens)
    elif mode == 'stagger':
        await run_staggered(fancy, max_tokens)
    elif mode == 'sequential':
        await run_sequential(fancy, max_tokens)
    else:
        await run_overlapping(fancy, max_tokens)
        await asyncio.sleep(3)
        await run_staggered(fancy, max_tokens)
        await asyncio.sleep(3)
        await run_sequential(fancy, max_tokens)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fancy', action='store_true')
    p.add_argument('--mode', choices=['overlap', 'stagger', 'sequential', 'all'], default='all')
    p.add_argument('--max-tokens', type=int, default=80)
    args = p.parse_args()
    asyncio.run(main(args.mode, args.fancy, args.max_tokens))