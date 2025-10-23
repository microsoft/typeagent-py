# How to Reproduce the Demos

## How we did the Monty Python demo

The demo consisted of loading a number (specifically, 11) popular
Monty Python sketches in a database, and asking questions about them.
The loading (ingestion) process was done ahead as it takes a long time.

The sketches were taken from
[ibras.dk](https://ibras.dk/montypython/justthewords.htm)
and converted to [WebVTT format](https://en.wikipedia.org/wiki/WebVTT)
format with "voice annotations" (e.g. `<v Shopkeeper>It's resting</v>`)
with help from a popular LLM.

We have a driver program in the repo to ingest WebVTT files into a
SQLite database.

This is `tools/ingest_vtt.py`. You run it as follows:
```sh
python tools/ingest_vtt.py FILE1.vtt ... FILEN.vtt -d mp.db
```
(This requires [configuring](env-vars.md) an API key etc.)

The process took maybe 15 minutes for 11 sketches.

The sketches can now be queried by using another tool:
```sh
python tools/query.py -d mp.db
```
(You just type questions and it prints answers.)

## How we did the GMail demo

The demo consisted of loading a large number (around 500) email messages
into a database, and querying the database about those messages.
The loading (ingestion) process was done ahead as it takes a long time.

We used the GMail API to download 550 messages from Guido's GMail
(details below).

Given a folder with `*.eml` files in MIME format, we ran our email
ingestion tool, `tools/test_email.py`. (All these details will change
in the future, hopefully to be more similar to `ingest_vtt.py`.)

The tool takes one positional argument, a directory, in which it will
create a SQLite database named `gmail.db`.
```sh
python tools/test_email.py .
```
The tool is interactive. The only command to issue is the following:
```sh
@add_messages --path "email-folder"
```
The process took over an hour for 500 messages. Moreover, it complained
about nearly 10% of the messages due to timeouts or just overly large
files. When an error occurs, the tool recovers and continues with the
next file.

We can then query the `gmail.db` database using the same `query.py`
tool that we used for the Monty Python demo.

### How to use the GMail API to download messages

In the `gmail/` folder you'll find a tool named `gmail_dump.py` which
will download any number of messages (default 50) using the GMail API.
In order to use the GMail API, however, you have to create a
Google Cloud app and configure it appropriately.

In order to figure out how to set up the (free) Google Cloud app we
used the instructions at [GeeksForGeeks
](https://www.geeksforgeeks.org/devops/how-to-create-a-gcp-project/).

The rest of the email ingestion pipeline doesn't care where you got
your `*.eml` files from -- every email provider has its own quirks.
`