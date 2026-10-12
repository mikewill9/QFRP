1. What the test says 

Amihud ranks stocks by illiquidity. It passed with p between 0.0005 and 0.005 at k=1 in periods 2005_2009, 2005_2024, 2010_2014, 2015_2019 and it failed in 2020 to 2024 (Sharpe 0.22, p = 0.29). The 2005_2009, 2010_2014 and 2015_2019 results were rerun at 20,000 shuffles with the same verdicts. 

2. What the test can and can't answer

It answers: "Is this ranking better than a random ranking within this universe?"
It can't answer: "Was this universe a fair set of stocks to choose from in 2005?"
Why shuffling can't fix it: The shuffle tests whether the ranking is linked to returns. It can't tell whether the link comes from the signal or from how the stocks were picked.

3. Why the universe is the problem

The 148 stocks came from about 150 of today's large caps, which were picked in summer 2026 (MMC was dropped due to faulty data, and EA was dropped because it was delisted, a small example of the kind of company a today's-winners list leaves out), leaving 148.
A stock that was relatively illiquid in 2005 and is still in the list today had to survive and grow into a large company by 2026 to be included on the list.
So the factor may be rewarded because the list only contains stocks that went on to succeed. Sharpe declines from 1.63 to 0.22 as the window nears the list date, which fits survivorship but also fits a fading premium. Since the list was built in 2026, tickers from 2020-2024 had much less time to drop out, and survivorship should matter least there. That is also the window where the factor fails (Sharpe 0.22).
Missing from the list: failures; bankruptcies, acquisitions, delistings, etc.

4. What would fix it 

A point-in-time universe that includes stocks that were tradable then but are gone now.

5. Demo takeaway line

Passing the test means the signal beat luck, not that the data was fair.