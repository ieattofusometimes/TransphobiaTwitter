import csv
from transformers import pipeline


def retrieve_word_appearance_count(file_path, word: str) -> int:
    # Returns the n of lines in a database that include a certain token.
    token_count = 0
    with open(file_path, encoding='utf-8') as file_csv:
        csv_reader = csv.reader(file_csv)
        for row in csv_reader:
            if word in row[0].lower():  # In our database the main text is in the first column of the database.
                token_count += 1
    return token_count


def retrieve_word_appearance_tweet(file_path: str, word: str) -> list:
    # Returns a list of tweets where the word appears.
    list_of_tweets = []
    with open(file_path, encoding='utf-8') as file_csv:
        csv_reader = csv.reader(file_csv)
        for row in csv_reader:
            sentence = row[0]
            if word in sentence.lower():
                list_of_tweets.append(sentence)
    return list_of_tweets


def main():
    # print(retrieve_word_appearance_count(tweets_reader, "vent"))
    model_path = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
    sentiment_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    some_tweets = retrieve_word_appearance_tweet('twitter.csv', 'lul')
    for tweet in some_tweets:
        print(tweet, sentiment_classifier(tweet))



if __name__ == "__main__":
    main()
