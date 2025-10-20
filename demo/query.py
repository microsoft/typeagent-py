from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage


async def main():
    conversation = await create_conversation("demo.db", TranscriptMessage)
    question = "Who volunteered to do the python library?"
    print("Q:", question)
    answer = await conversation.query(question)
    print("A:", answer)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
