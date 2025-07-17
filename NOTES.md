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
