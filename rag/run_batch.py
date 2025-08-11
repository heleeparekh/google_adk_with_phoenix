# run_batch.py — drive multiple CSV questions through the agentimport asyncio
import random
import pandas as pd
import asyncio
from rag.rag_agent_csv import run_rag  # re-use the function

CSV_PATH = "rag_sample_qas_from_kis.csv"

async def main(n: int = 10, seed: int = 7):
    df = pd.read_csv(CSV_PATH)
    # pick n random rows (or slice/filter however you like):
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    idxs = idxs[:n]

    for i in idxs:
        q = str(df.loc[i, "sample_question"])
        print(f"\n=== Q{ i } === {q}")
        ctx, ans = await run_rag(q, query_id=str(i))
        print("---- Context (truncated) ----")
        print(ctx[:500] + ("..." if len(ctx) > 500 else ""))
        print("---- Answer ----")
        print(ans)

if __name__ == "__main__":
    asyncio.run(main(n=12))
