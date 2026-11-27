# How much signing work does one more chain buy?

For every 256-bit digest, suppose verification can spend at most 442 forward chain steps. With 66 signature elements, the smallest possible total number of public chain steps is 990. With 67 elements it is 953. A code attaining the second bound uses 52 chains of length 14 and 15 of length 15. Every digest costs 511 steps to sign and 442 to verify. The extra signature element therefore saves exactly 37 signing steps when comparing the optimal worst-case signers at this verifier budget.

## Model and exact capacity

Choose public integer chain lengths $h_i\ge 0$ for $1\le i\le n$, measured in hash applications from seed to endpoint. The lengths are fixed across all digests. Write $H=\sum_i h_i$. Let $x_i$ be the number of forward steps *remaining* after revealing a chain element. Verification costs $|x|=\sum_i x_i$ and signing from the seeds costs $H-|x|$. The code injects the entire digest space into pairwise coordinatewise-incomparable vectors. The sign convention reverses every coordinate together, preserving incomparability.

For $n\ge 1$ and integer budgets $H,V\ge 0$, let $C(n,H,V)$ be the largest number of digests possible over *all* such endpoint allocations of total length $H$ with $|x|\le V$ on every digest. Set $H=nq+s$, $0\le s<n$, $t=\min(V,\lfloor H/2\rfloor)$ and $P_j(z)=1+z+\cdots+z^j$. Then

$$
C(n,H,V)=[z^t] P_q(z)^{n-s}P_{q+1}(z)^s .
$$

**For fixed endpoints.** The product of finite chains has a decomposition into saturated symmetric chains (de Bruijn, van Ebbenhorst Tengbergen and Kruyswijk). A chain starting at rank $a$ ends at rank $H-a$. Any such chain meeting the verifier budget crosses level $t$: when $V$ is below the middle its starting rank is at most $V$, and otherwise every chain crosses the middle. An antichain takes at most one member per chain. The members of level $t$ attain the bound, since distinct vectors of the same sum are incomparable. Its size is the coefficient of $z^t$ in $\prod_i P_{h_i}$.

**Optimizing the endpoints.** Take two lengths $a\ge b+2$. Moving a step from the longer to the shorter chain changes the two-factor polynomial by

$$
P_{a-1}P_{b+1}-P_aP_b=z^{b+1}+z^{b+2}+\cdots+z^{a-1}.
$$

This follows by canceling the common product or expanding the geometric sums. All other factors have nonnegative coefficients. Consequently this exchange can only increase the size of every rank simultaneously. The sum of squared lengths strictly decreases by $2(a-b-1)$, so finitely many exchanges lead to the balanced multiset in the formula. This proves global optimality even if the alternative code uses multiple ranks or has chains longer than 15. The balanced allocation is a maximizer. Ties at an individual rank are possible.

**Testing minimality.** Capacity is nondecreasing in $H$: extend a single endpoint by one in residual coordinates and the old box embeds with the same ranks and order. Thus failure at $H-1$ and success at $H$ certify the first feasible budget. For rows with $t=V$, every admissible geometry must sign from seed for at least $H_{\min}-V$ steps on every digest. The displayed fixed-rank code attains this bound on each digest, also solving the minimax signing problem.

## Integer certificates for 256-bit digests

“Middle” removes the verifier cap and reports a central-layer witness. The other rows give both the minimum endpoint budget and a signer attaining the lower bound at the requested cap.

| Elements | Verifier cap | Minimum total steps | Length 14 | Length 15 | Witness verify | Witness sign |
|---:|---:|---:|---:|---:|---:|---:|
| 66 | 442 | 990 | 0 | 66 | 442 | 548 |
| 67 | 442 | 953 | 52 | 15 | 442 | 511 |
| 66 | 450 | 985 | 5 | 61 | 450 | 535 |
| 67 | 450 | 950 | 55 | 12 | 450 | 500 |
| 66 | 460 | 980 | 10 | 56 | 460 | 520 |
| 67 | 460 | 947 | 58 | 9 | 460 | 487 |
| 66 | Middle | 975 | 15 | 51 | 487 | 488 |
| 67 | Middle | 946 | 59 | 8 | 473 | 473 |

For the 67-element, 442-step case, the two capacities bracketing $2^{256}$ are

```text
H = 952   110182451069593983686688005416603323146792524836745351781989564094751627569638
2^256     115792089237316195423570985008687907853269984665640564039457584007913129639936
H = 953   115974534170143938072359597389303066128773213585941303278612403016409153623718
```

This Python 3 block checks both sides of every table row in two ways. The first method multiplies polynomials truncated above the target rank. The second extracts the same coefficient from $(1-z^{q+1})^{n-s}(1-z^{q+2})^s/(1-z)^n$ by inclusion–exclusion.

```python
from math import comb

def capacity_pair(n, total, cap):
    q, s = divmod(total, n)
    t = min(cap, total // 2)
    counts = [1] + [0] * t
    for length in [q] * (n - s) + [q + 1] * s:
        new = []
        window = 0
        for k, count in enumerate(counts):
            window += count
            if k > length:
                window -= counts[k - length - 1]
            new.append(window)
        counts = new

    alternating = 0
    for i in range(n - s + 1):
        for j in range(s + 1):
            k = t - (q + 1) * i - (q + 2) * j
            if k >= 0:
                alternating += ((-1) ** (i + j) * comb(n - s, i)
                                * comb(s, j) * comb(k + n - 1, n - 1))
    return counts[t], alternating

rows = [(66, 442, 990), (67, 442, 953), (66, 450, 985),
        (67, 450, 950), (66, 460, 980), (67, 460, 947),
        (66, 1000, 975), (67, 1000, 946)]
for n, cap, total in rows:
    lo, lo_alt = capacity_pair(n, total - 1, cap)
    hi, hi_alt = capacity_pair(n, total, cap)
    assert lo == lo_alt and hi == hi_alt
    assert lo < 2**256 <= hi
    print(n, cap, total, lo, hi)
```

## Interpretation, implementation cost and limits

The uniform 66-chain, 15-step geometry in track PR #7 survives a stronger comparison: at a verifier cap of 442, no redistribution can reduce the 990-step endpoint total. Keeping 66 elements but raising the cap to 450 saves 13 signing steps, of which eight move to verification and five disappear from the endpoint total. At 32 bytes per chain element, adding a chain increases the chain payload from 2,112 to 2,144 bytes. The latter matches the WOTS+ chain payload at $n=32,w=16$ in RFC 8391, §3.1. The encoders differ. The sum $H$ also gives the number of steps to generate all endpoints from seeds in this model.

To encode, precompute suffix rank counts through $t$. Treat the digest as an integer in $[0,2^{256})$, partition the current rank by the first digit and subtract earlier bucket sizes until the bucket containing the index is reached. Repeat on the suffix. The bucket sizes sum to the current count, so this defines an injection by induction. A sliding window builds the suffix table in $O(n(t+1))$ integer operations and cells; a linear scan then uses at most $\sum_i(h_i+1)$ bucket inspections per digest. These are big-integer operations. The cost model excludes the encoder, address generation, Merkle authentication, signature parsing, aggregation proof constraints and hardware parallelism. It gives neither an end-to-end speed estimate nor a quantum unforgeability theorem. Randomized or collision-allowing encodings have different feasible sets.

## Prior art and scope of the contribution

De Bruijn, van Ebbenhorst Tengbergen and Kruyswijk supply the symmetric-chain theorem. Zhang, Cui and Yu analyze constant-sum Winternitz encodings. Khovratovich, Kudinov and Wagner study verification cost and alternative encodings, including uniform-base parameter sweeps in §5. The track's PR #7 analyzes the fixed geometry $(15,\ldots,15)$. The contribution here is the reduction from freely allocated endpoints at fixed total to one balanced coefficient and the reproducible strict thresholds on both sides of eight parameter choices. The argument integrates classical facts; the literature comparison establishes no general priority for coefficientwise balancing.

- [On the set of divisors of a number (1951)](https://research.tue.nl/en/publications/on-the-set-of-divisors-of-a-number/)
- [Revisiting the Constant-sum Winternitz One-time Signature (2023/850)](https://eprint.iacr.org/2023/850)
- [At the Top of the Hypercube (2025/889)](https://eprint.iacr.org/2025/889)
- [Fixed 66-chain verification budget, PR #7](https://github.com/MuncleUscles/post-quantum-signatures/pull/7)
- [WOTS+ parameters and algorithms, RFC 8391](https://www.rfc-editor.org/rfc/rfc8391#section-3.1)
