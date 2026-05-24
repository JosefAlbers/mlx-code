from mlx_code import mcb_tool
from mlx_code.repl import run_repl

if __name__ == "__main__":
    dt = mcb_tool.DocThread()
    dt.submit(
        "\n    Lamport, Shostak, and Pease (1982) introduced Byzantine fault tolerance (BFT) in their paper\n    'Byzantine Generals Problem'. They showed that a system of n nodes can tolerate at most f\n    Byzantine (arbitrarily malicious) faults if n >= 3f+1. The paper introduced oral and signed\n    message protocols and proved tight lower bounds on the number of rounds required.\n    "
    )
    dt.submit(
        "\n    Castro and Liskov (1999) introduced PBFT (Practical Byzantine Fault Tolerance), the first\n    BFT protocol efficient enough for practical use. It operates in O(n^2) message complexity\n    per consensus round and requires n >= 3f+1 replicas. PBFT introduced the prepare-commit\n    two-phase protocol that became the template for most subsequent BFT systems.\n    "
    )
    dt.submit(
        "\n    HotStuff (Abraham et al. 2018) achieves linear message complexity O(n) per round using a\n    leader-based protocol with threshold signatures. It introduced the concept of chained phases\n    where each vote simultaneously finalises the previous block and votes on the current one.\n    HotStuff is the basis of the consensus layer in several production blockchains including\n    Facebook's Diem (formerly Libra).\n    "
    )
    dt.submit(
        "\n    Tendermint (Buchman 2016) is a BFT consensus protocol designed for blockchain use. It uses\n    a rotating leader and a two-phase (prevote/precommit) protocol. Unlike PBFT it is designed\n    for open networks and introduces the concept of validators with stake-weighted voting.\n    Safety is guaranteed under asynchrony; liveness requires partial synchrony.\n    "
    )
    dt.submit(
        "\n    A key open problem in BFT is the scalability of the validator set. Most classical protocols\n    degrade at n > 100 due to O(n^2) or O(n) all-to-all messaging. Recent approaches include\n    committee sampling (Algorand), sharding, and recursive proof aggregation (SNARKs) to amortise\n    the cost of Byzantine agreement across large validator sets.\n    "
    )
    run_repl(
        system=mcb_tool.system_prompt(dt),
        init_prompt="Write a synthesis on how BFT consensus protocols have evolved in terms of message complexity.",
        extra_tool_classes=mcb_tool.ALL_TOOLS,
        tool_names=mcb_tool.ALL_NAMES,
        ctx={"dt": dt},
        sdir=".",
    )
