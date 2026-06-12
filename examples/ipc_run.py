import asyncio, pathlib
from mlx_code.repl import Agent

async def pipe(system='You are a research assistant. Always write final output to the file specified.', draft='draft.md', final='final.md', kb='kb'):
    pathlib.Path(kb).mkdir(parents=True, exist_ok=True)
    agent_a = Agent(system)
    await agent_a.run(f'Research the history of Byzantine fault tolerance. Write a structured summary with sections: Background, Key Papers, Open Problems. Save it to `{kb}/{draft}`. Nothing else.')
    agent_b = Agent(system)
    await agent_b.run(f'Read `{kb}/{draft}` in full. Using ONLY information in that file (no external knowledge), extract a JSON list of all named papers with keys: title, authors, year, contribution. Write the JSON to `{kb}/{final}`. Do not add anything not in the file.')

async def fork_and_wait(kb='kb'):
    pathlib.Path(kb).mkdir(parents=True, exist_ok=True)
    system = 'You are a research assistant. Always write final output to the file specified.'
    topics = ['history', 'algorithms', 'industry_usage']
    tasks = []
    for topic in topics:
        agent = Agent(system)
        task = agent.run(f'Research the {topic} of Byzantine Fault Tolerance. You MUST use the Write tool to save findings to `{kb}/{topic}.md`. Do not just report the findings back.')
        tasks.append(task)
    print(f'Spawning {len(tasks)} workers...')
    await asyncio.gather(*tasks, return_exceptions=True)
    print('All workers returned. Check the folder now.')
    reducer = Agent(system)
    await reducer.run(f'Read all files in `{kb}/`. Synthesize into `final_report.md`.')
if __name__ == '__main__':
    asyncio.run(pipe())