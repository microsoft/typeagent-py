# How to Reproduce the Demos

All demos require [configuring](env-vars.md) an API key etc.

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

The process took maybe 15 minutes for 11 sketches.

The sketches can now be queried by using another tool:

```sh
python tools/query.py -d mp.db
```

(You just type questions and it prints answers.)

## How we did the Gmail demo

The demo consisted of loading a large number (around 500) email messages
into a database, and querying the database about those messages.
The loading (ingestion) process was done ahead as it takes a long time.

We used the Gmail API to download 550 messages from Guido's Gmail
(details below).

Given a folder with `*.eml` files in MIME format, we ran our email
ingestion tool, `tools/ingest_email.py`. You run it as follows:

```sh
python tools/ingest_email.py -d gmail.db email-folder/
```

You can also pass individual `.eml` files instead of a directory.
Use `-v` for verbose output.

### Filtering by date

Use `--start-date` and `--stop-date` to restrict ingestion to a date range with [start, stop):

```sh
# Ingest only January 2024 emails
python tools/ingest_email.py -d gmail.db email-folder/ \
    --start-date 2024-01-01 --stop-date 2024-02-01
```

### Pagination with --offset and --limit

These flags slice the input file list before any other filtering:

```sh
# Ingest only the first 20 files
python tools/ingest_email.py -d gmail.db email-folder/ --limit 20

# Skip the first 100 files, then process the next 50
python tools/ingest_email.py -d gmail.db email-folder/ \
    --offset 100 --limit 50
```

All four flags can be combined. The filter pipeline is:
offset/limit → already-ingested → date range.

The process took over an hour for 500 messages. Moreover, it complained
about nearly 10% of the messages due to timeouts or just overly large
files. When an error occurs, the tool recovers and continues with the
next file. Previously ingested emails are automatically skipped on
subsequent runs.

We can then query the `gmail.db` database using the same `query.py`
tool that we used for the Monty Python demo:

```sh
python tools/query.py -d gmail.db
```

### How to use the Gmail API to download messages

In the `tools/mail/` folder you'll find a tool named `gmail_dump.py` which
will download any number of messages (default 50) using the Gmail API.
In order to use the Gmail API, however, you have to create a (free)
Google Cloud app and configure it appropriately.

We created  created an app in test mode at
[Google Cloud Console](https://console.cloud.google.com) and gave it
access to the Gmail API (I forget how exactly we did this part).

To create the needed client secret, we navigated to Client (side bar)
and clicked on "+ Create Client" (in the row of actions at the top),
selected "Desktop app", gave it a name, hit Create, scrolled down in the
resulting dialog box, and hit "Download JSON". This produced a JSON file
which should be copied into _client_secret.json_ in the gmail folder.
(The Cloud Console interface may look different for you.)

The first time you run the gmail_dump.py script, it will take you to
a browser where you have to log in and agree to various warnings about
using an app in test mode etc. The gmail_dump.py script then writes a
file _token.json_ and you're good for a week or so. When token.json
expires, unfortunately you get a crash and you have to manually delete
it to trigger the login flow again.
(Sometimes starting a browser may fail, e.g. under WSL. Take the URL
that's printed and manually go there.)

The rest of the email ingestion pipeline doesn't care where you got
your `*.eml` files from -- every email provider has its own quirks.

## Bonus content: Podcast demo

The podcast demo is actually the easiest to run:
The "database" is included in the repo as
`tests/testdata/Episode_53_AdrianTchaikovsky_index*`,
and this is in fact the default "database" used by `tools/query.py`
when no `-d`/`--database` flag is given.

This "database" indexes `tests/testdata/Episode_53_AdrianTchaikovsky.txt`.
It was created by a one-off script that invoked
`src/typeagent/podcast/podcast_ingest/ingest_podcast()`
and saved to two files by calling the `.ingest()` method on the
returned `src/typeagent/podcasts/podcast/Podcast` object.

Here's a brief sample session:

```sh
$ python tools/query.py
1.318s -- Using Azure OpenAI
0.054s -- Loading podcast from 'tests/testdata/Episode_53_AdrianTchaikovsky_index'
TypeAgent demo UI 0.2 (type 'q' to exit)
TypeAgent> What did Kevin say to Adrian about science fiction?
--------------------------------------------------
Kevin Scott expressed his admiration for Adrian Tchaikovsky as his favorite science fiction author. He mentioned that Adrian has a new trilogy called The Final Architecture, and Kevin is eagerly awaiting the third book, Lords of Uncreation, which he has had on preorder for months. Kevin praised Adrian for his impressive writing skills and his ability to produce large, interesting science fiction books at a rate of about one per year.
--------------------------------------------------
TypeAgent> How was Asimov mentioned.
--------------------------------------------------
Asimov was mentioned in the context of discussing the ethical and moral issues surrounding AI development. Adrian Tchaikovsky referenced Asimov's Laws of Robotics, noting that Asimov's stories often highlight the inadequacy of these laws in governing robots.
--------------------------------------------------
TypeAgent> q
$
```

Enjoy exploring!
