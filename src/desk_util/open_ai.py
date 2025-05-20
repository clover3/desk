import json
import logging
import os

import openai
from openai import OpenAI
from openai.lib.azure import AzureOpenAI

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import data_root_path
from rule_gen.cpath import output_root_path


def get_state_db_path():
    p = os.path.join(output_root_path, "state", "openai_use.db")
    make_parent_exists(p)
    return p


def setup_file_logger(logger_name):
    log_file = os.path.join(output_root_path, "log", "{}.log".format(logger_name))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


import sqlite3


class TokenUsageDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = get_state_db_path()
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database and create table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT UNIQUE,
                    tokens_used INTEGER
                )
            ''')
            conn.commit()

    def add_usage(self, model, tokens_used):
        """Add to or update the token usage for a specific model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if an entry already exists for this model
            cursor.execute('SELECT tokens_used FROM token_usage WHERE model = ?', (model,))
            row = cursor.fetchone()

            if row:
                # If entry exists, update the token count by adding the new tokens_used
                current_tokens = row[0]
                new_total_tokens = current_tokens + tokens_used
                cursor.execute('''
                    UPDATE token_usage 
                    SET tokens_used = ? 
                    WHERE model = ?
                ''', (new_total_tokens, model))
            else:
                # If no entry exists, insert a new row
                cursor.execute('''
                    INSERT INTO token_usage (model, tokens_used) 
                    VALUES (?, ?)
                ''', (model, tokens_used))

            conn.commit()

    def get_all_usages(self):
        """Retrieve all records of token usage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM token_usage')
            rows = cursor.fetchall()
        return rows


import time
import openai

class OpenAIEmptyResponse(ValueError):
    pass


def complete_with_retry(client, model, messages, max_retries=5, sleep_seconds=20, logprobs=None, top_logprobs=None):
    initial_sleep = sleep_seconds
    for attempt in range(max_retries):
        try:
            # Add logprobs parameters to the request
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
            if not chat_completion.choices:
                raise OpenAIEmptyResponse()

            return chat_completion
        except (openai.RateLimitError, OpenAIEmptyResponse) as e:
            if attempt == max_retries - 1:
                raise Exception("Max retries exceeded")

            sleep_duration = initial_sleep * (2 ** attempt)  # Exponential backoff
            print(
                f"{str(e)}. Retrying in {sleep_duration} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(sleep_duration)

    print("OpenAI request failed")
    raise openai.RateLimitError


class OpenAIChatClient:
    def __init__(self, model="gpt-4o", db_client=None, logprobs=False, top_logprobs=1):
        self.logger = setup_file_logger("openai")
        self.client = get_open_ai()
        self.model = model
        self.total_tokens_used = 0  # Track total tokens used
        self.db_client = TokenUsageDB() if db_client is None else db_client
        self.logprobs = logprobs  # Whether to request token probabilities
        self.top_logprobs = top_logprobs  # Number of top probabilities to return

    def request(self, message):
        chat_completion = complete_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": message}],
        )
        # Track tokens used
        try:
            last_request_tokens = chat_completion.usage.total_tokens
            self.total_tokens_used += last_request_tokens
        except openai.OpenAIError as e:
            raise e
        except AttributeError as e:
            print("chat_completion", chat_completion)
            raise e

        # Extract token probabilities if available
        token_probs = None
        if self.logprobs and hasattr(chat_completion.choices[0], 'logprobs'):
            token_probs = chat_completion.choices[0].logprobs

        # Log message with tokens and token probabilities
        msg = {
            "request": message,
            "response": chat_completion.choices[0].message.content,
            "tokens_used": last_request_tokens,
            "total_tokens_used": self.total_tokens_used
        }

        # Add token probabilities to the log if available
        if token_probs:
            msg["token_probs"] = token_probs.to_dict() if hasattr(token_probs, 'to_dict') else token_probs

        self.logger.info(json.dumps(msg))

        # Store token probability data in database if enabled

        return chat_completion.choices[0].message.content

    def request_with_probs(self, message):
        """Return both the response content and the token probabilities."""
        chat_completion = complete_with_retry(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": message}],
            logprobs=True,
            top_logprobs=self.top_logprobs
        )

        # Track tokens used
        try:
            last_request_tokens = chat_completion.usage.total_tokens
            self.total_tokens_used += last_request_tokens
        except (openai.OpenAIError, AttributeError) as e:
            print("Error tracking tokens:", e)

        # Extract token probabilities
        token_probs = None
        if hasattr(chat_completion.choices[0], 'logprobs'):
            token_probs = chat_completion.choices[0].logprobs

        # Log message with tokens and probabilities
        msg = {
            "request": message,
            "response": chat_completion.choices[0].message.content,
            "tokens_used": last_request_tokens if 'last_request_tokens' in locals() else None,
            "total_tokens_used": self.total_tokens_used
        }

        if token_probs:
            msg["token_probs"] = token_probs.to_dict() if hasattr(token_probs, 'to_dict') else token_probs

        self.logger.info(json.dumps(msg))

        # Store data in database
        token_probs = [(item.token, item.logprob) for item in token_probs.content]

        return {
            "content": chat_completion.choices[0].message.content,
            "token_probs": token_probs
        }

    def __del__(self):
        """Update the database with total token usage upon destruction."""
        print("{} tokens used.".format(self.total_tokens_used))
        if self.db_client:
            self.db_client.add_usage(self.model, self.total_tokens_used)


def get_open_ai_my():
    api_key_path = os.path.join(data_root_path, "openai_api_key.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()

    return OpenAI(api_key=key)


def get_open_ai(deployment=None):
    api_key_path = os.path.join(data_root_path, "openai_api_key_uva.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()
    azure_endpoint = "https://rtp2-shared.openai.azure.com"
    client = AzureOpenAI(api_key=key,
                         api_version="2024-10-21",
                         azure_endpoint=azure_endpoint, azure_deployment=deployment)
    return client



def get_open_ai_2():
    api_key_path = os.path.join(data_root_path, "openai_api_key_uva2.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()
    endpoint = "https://rtp2-gpt35.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    client = AzureOpenAI(api_key=key, api_version="2024-10-21",
                         azure_endpoint=endpoint, azure_deployment=None)
    return client


class DummyChatClient:
    def __init__(self):
        self.total_tokens_used = 0  # Track total tokens used
        self.last_request_tokens = 0  # Track tokens used in the last request

    def request(self, message):
        char_per_token = 4

        self.last_request_tokens = int(len(message) // char_per_token)
        self.total_tokens_used += self.last_request_tokens
        return ""

    def __del__(self):
        print("{} tokens used.".format(self.total_tokens_used))


def main():
    client = get_open_ai()
    # client = get_open_ai_2()
    texts = ["Fantastic appearance: flawless chest of the second size, wasp waist, well-groomed skin, attractive features. It is interesting to talk with me and hot in bed. I will become an adornment of fashionable parties and business negotiations, and the fact that I can get up in bed will drive you crazy with pleasure. Most Relevant Video Results: \"twintail hentai\"\n\nTag: gonna be the twin-tails - <Organization>\n\nLogin Sign Up. Login to your account Remember Me Register a new account Lost your password? One day, while doing his rounds through the school he was attending, <Person> is attacked by a never before seen creature that almost makes him lose Caught sneaking inside the wrong part of town, this cute ginger babe is met with a completely unexpected faith! Jumping out of nowhere comes a In spite of having almost all of her life spent as a very well-behaving girl, this cutie now craves for some intensity and variety in her life\n\n\n\n<Person>. Age: 25. Do you need to hide from the outside world and satisfy in bed ?! Oh, with me it is not enough that it is possible, I also very much want it. You will swim in the oceans and seas of my passion and tenderness! Thumbnails\n\nMom pov anal creampie Muvis porn Unnatural sex 20 Best young porn videos Sexy legs and ass tumblr Scarlett <Person> nude in a good woman Monsters of cock gif Drunk passed naked girls. Gonna be the twin tail hentai. Share this video:. You need the latest version of Adobe Flash Player to view this video. Click here to download. Offering exclusive content not available on Pornhub. The <<Organization>> team is always updating and adding more porn videos every day.", "Gorgeous Blonde Lady On Real Homemade\n\nSlim Body Beauty Footjob And Ass Fucking Tube Porn Video\n\nEating his own cum off that guys. Popular fashionably laid scene 3 - <Person> sex. Free download homemade sex clips and also porn jumping nude girl pictures free homemade movie nudist prom woman. Adult friend finder alternative names and also 15 man cum swallow. Best pornstar <Person> in fabulous small tits, facial porn movie. En pblic uniforme asitica japonesa mamada en pblic en pblic asitica. Sex and submission pair of sexy sluts caught in the woods and punished by pervert man in. Glamour legal age teenager riding old mans pole. Sexy girl shows off seen on stupidcamscom. <Person> gladly showed off his cock sucking skills by swallowing <Person>'s cock deep right down to his nuts only to do it again and again. Naked thailand lady on the floor in her fuzzy heels. <Person> made a circle of his thumb and forefinger and moved it up and down his brothers prick so that he only agitated the rim around the helmet. Inmate fucks his busty blonde. College teens jerking in dorm before facial voyeur porn.\n\nWatch free cheating wives at strip club videos at heavyr, women out of control at male strip club. cheating wife caught on babysitter cam. wife cheating at male strip club. Amazing all natural and sweet office lady <Person> was caught by her ex boyfriend right on her workplace.", "porn video HD 323 Model futa spitroast double blowjob\n\n<Person> double blowjob mtf fisting\n\nStep sister is horny and wants not her step brother 13,, Real bruised ass in this one. Whips his girlfriend whip Tags: mature , lesbian , spanking , girlfriend , fetish Duration: 5 mins Added: 1 month ago. Tags: mom , milf , lesbian , spanking , mother , taboo , threesome , group sex , orgy , redhead. Top Trending Playlists View More. Pornstars: sabrina rose. <Person>. Age: 26. I am the most regarded delightful and highly discreet independent courtesan, The service that I provide goes beyond expectations\n\nNew Videos by <Organization>\n\n\n\nPorn Images Double blowjob fishnet mmf softcore\n\nSport fishnet double blowjob mmf\n\n'lesbian fishnet' Search - PSSUCAI.INFO\n\n\n\nMature Pauline, 21 y. i am very interesting,natural and magic young lady ab"]

    message = texts[1][:300]
    print(message)
    message = "summarize: " + message
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    print(chat_completion.choices[0].message.content)


def test_log_prob():
    client = OpenAIChatClient(logprobs=True)
    ret = client.request_with_probs("Hi")
    print(ret)



if __name__ == "__main__":
    test_log_prob()
