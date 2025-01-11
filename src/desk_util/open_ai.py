import json
import logging
import os

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

    file_handler = logging.FileHandler(log_file)
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


class OpenAIChatClient:
    def __init__(self, model="gpt-4o", db_client=None):
        self.logger = setup_file_logger("openai")
        self.client = get_open_ai()
        self.model = model
        self.total_tokens_used = 0  # Track total tokens used
        self.last_request_tokens = 0  # Track tokens used in the last request
        self.db_client = TokenUsageDB()  # Accepts an instance of TokenUsageDB

    def request(self, message):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )

        # Track tokens used
        self.last_request_tokens = chat_completion.usage.total_tokens
        self.total_tokens_used += self.last_request_tokens

        # Log message with tokens
        msg = {
            "request": message,
            "response": chat_completion.choices[0].message.content,
            "tokens_used": self.last_request_tokens,
            "total_tokens_used": self.total_tokens_used
        }
        self.logger.info(json.dumps(msg))

        return chat_completion.choices[0].message.content

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
    client = AzureOpenAI(api_key=key, api_version="2024-10-21",
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


if __name__ == "__main__":
    main()
