Notes
=====

Weights are slightly off in many places
---------------------------------------

In particular the weights computed by TS are slightly different from
those computed by Python from the same LLM-produced query.

- Tried using double precision for dot products between two vectors.
  - This showed differences in the 6th or 7th digit, whereas what we
    are seeing is in the 2nd or 3rd digit. So more precise dot product
    computation is not the fix we'd hoped for.

- Concluding, this is **STILL OPEN**.

Warnings about missing message/semref ordinals
-----------------------------------------------

When comparing lists of message ordinals:

- Messages 42, 46, 52, 68, 70 are all just Kevin saying "Yeah."
  So those differences are irrelevant (false positives).
  - **Fixed by excluding them from the warning.**

- Concluding, the only question that's got a true warning about the
  query result is 55, "How long did Adrian struggle before he got
  published?"
  - **Fixed a bug in how we were calling intersect().**
  - Now, the result is worse until I implement fallbacks.

- That question also gets an answer that feels correct if verbose,
  but is scored "low" (0.885), probably due to that verbosity.
  This is certainly due to the answers.py logic/prompts.

- Steve suggests throwing noise text like "Yeah" out completely.
  Umesh replies that there's support for stop words etc.
  but the Podcast import doesn't use it.

Question 55
-----------

Looking at question 55 ("How long did Adrian struggle before
he got published?") we see 8 messages missing (none "noise"),
and many entities, actions and topics missing from the knowledge
results (which would explain the missing messages).

See above -- **fixed by fixing two intersect() calls.**

Questions 5 and 6
-----------------

These are variants that look for books in the first 15 minutes.
The query compilation seems to be off here.

**Fixed date/time conversions to account for proper timezones.**

We still see failures on 5, but they are hard to repro. Maybe this
hit a random network or server failure?

**Fixed by adding create_relevant_knowledge()**

Skynet
------

Six questions still have mismatching message or semref ordinals
when using precomputed query or compiled query.

- Five of these are all the questions about Skynet.
- Number six is 55 above (Adrian's struggle to get published).

The five Skynet questions are:

- 38 "What did Adrian say about Skynet?"
- 39 "Adrian talked about Skynet. What did he say?"
- 40: "What was Adrian's sentiment when he talked about Skynet?"
- 54: "What points did they make about Skynet?"
- 58: "What did Christina say about Skynet?"

**Solved by implementing fallback queries**

Remaining issues due to answer generator ("phase 4")
----------------------------------------------------

- 45 "Summarize Adrian's thoughts to Kevin?" gives no answer about half
  of the time, scoring alternatingly (or in short runs) 0.000 and 0.9xx.
  What is it about this query that triggers such strong stochasticity?

Examples of questions where Python's answer generator does better
-----------------------------------------------------------------

I don't do an exhaustive check but I didn't find any where TS did better.
Possibly this can be explained by the TS version's stingy character budget.
Python (yet) use an infinite character budget. (GPT-4o has a very large
token buffer limit, IIRC our entire podcast would fit in it.)
*Arguably Python is cheating here.*

- 7, 11 "List all books that are also movies" [Note that this question is repeated!]
  - TS: The Circle
  - Py: The books that are also movies mentioned in the provided context are:

        1. The Circle
  - Here Py adds a clarification because the question was ambiguous
    (there are of course many thousand books that are also movies).

- 17 "When did we talk about Profiles of the Future?"
  - TS: We talked about Profiles of the Future on May 1, 2023 at 07:53:21.
  - Py: We talked about 'Profiles of the Future' on May 1, 2023,
        at 07:53:21 UTC. Kevin Scott mentioned that Arthur C. Clarke wrote this book
        in 1968-1969, discussing the challenges of predicting the future.
  - Arguably better because it gives more detail (including the timezone).

- 20 "How did Asimov get referenced?"
  - TS: Asimov was referenced by Adrian Tchaikovsky on May 1, 2023 at 07:37:18.402Z.
  - Py: Asimov was referenced in the context of discussing the ethical and
        philosophical issues surrounding AI. Adrian Tchaikovsky mentioned
        Asimov's Laws of Robotics, highlighting that Asimov's stories often
        pointed out the inadequacy of these laws to govern robots effectively.
  - TS mistreats "how" as "when". Py is all round more informative.

- 29, 57 "Where did Adrian study?" [Note that this question is repeated!]
  - TS: Adrian Tchaikovsky studied zoology and psychology.
  - Py: Adrian studied zoology and psychology at the University of Reading.
  - TS doesn't even answer the "where" question!

- 54 "What points did they make about Skynet?"
  - TS: Skynet is described as an AI system with the intent to destroy humanity.
  - Py: Adrian Tchaikovsky discussed Skynet in the context of highly powerful
        AI systems. He mentioned that traditional sci-fi scenarios often depict
        Skynet as deciding to destroy humanity because its existence is threatened
        or it needs resources. He also noted that Skynet's desire not to be turned
        off makes sense to humans but not necessarily to Skynet as a computer,
        which wouldn't have an innate drive to preserve its own existence unless
        programmed to do so.
  - Py gives much more color in its answer -- TS barely gives one point.

Remaining issues due to query generator ("phase 1")
---------------------------------------------------

- 5: "List all book titles from the first 15 minutes"
  - Consistently no answer.
  - Likely cause: TS gives times in UTC; Py gives local times (PDT)

- 6 "List all book titles mentioned in the first 15 minutes"
  - Same as 5.

- 46 "Summarize Adrian's ideas about the Terminator?"
  - Py adds lots of search terms, including "person", "concept" and many more.

- 27: "Give me an overview of Portids"
  - Stochastic: 4 out of 5 get no answer.
  - Probably cause: search query is empty,
    should have `search_terms=["Portids"]` but has `search_terms=[]`.

- 43: "Summarize Kevin's thoughts on Artificial Intelligence?"
  - Consistent failure.

- 44: "Summarize Kevin's thoughts on AI?"
  - Consistent failure.

- 45: "Summarize Adrian's thoughts to Kevin?"
  - Stochastic, 2/5 fail to get an answer.
  - Suspicious: added type=["person"] to several filter values.

- 54: "What points did they make about Skynet?"
  - Consistent failure.

- 55: "How long did Adrian struggle before he got published?"
  - Consistent failure.
