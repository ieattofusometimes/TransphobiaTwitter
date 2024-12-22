import csv
# from transformers import pipeline


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


def retrieve_all_tweets(file_path: str) -> list:
    list_of_tweets = []
    with open(file_path, encoding='utf-8') as file_csv:
        csv_reader = csv.reader(file_csv)
        for row in csv_reader:
            sentence = row[0]
            list_of_tweets.append(sentence)
    return list_of_tweets


def calculate_precision(file_path: str, label: str) -> tuple:
    n_in_class = 0
    n_in_class_and_true = 0
    with open(file_path, encoding='utf-8') as file_csv:
        csv_reader = csv.reader(file_csv)
        for row in csv_reader:
            label_assigned = row[0].split(':')[1]
            value = row[0].split(':')[2]
            if label_assigned == label:
                n_in_class += 1
            if value == '"T"' and label_assigned == label:
                n_in_class_and_true += 1
    return n_in_class_and_true / n_in_class, n_in_class_and_true, n_in_class


def calculate_accuracy(file_path: str) -> tuple:
    total = 0
    true = 0
    with open(file_path, encoding='utf-8') as file_csv:
        csv_reader = csv.reader(file_csv)
        for row in csv_reader:
            total += 1
            value = row[0].split(':')[2]
            if value == '"T"':
                true += 1
    return true/total, true, total


def main():
    # print(retrieve_word_appearance_count(tweets_reader, "vent"))
    # toxicity_model_path = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
    # model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    # sentiment_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    # toxicity_classifier = pipeline("text-classification", model=toxicity_model_path, tokenizer=toxicity_model_path)
    # some_tweets = retrieve_word_appearance_tweet('twitter.csv', 'vrouw')
    all_tweets = retrieve_all_tweets('twitter.csv')
    # with open('results_toxicity.csv', 'w', encoding='utf-8') as file:
    #     for tweet in all_tweets:
    #         file.write('"' + tweet + '"')
    #         file.write(':' + '"' + toxicity_classifier(tweet)[0]['label'] + '"')
    #         file.write('\n')
    print(calculate_precision('results_toxicity.csv', '"Positive"'))
    print(calculate_precision('results_toxicity.csv', '"Negative"'))
    print(calculate_precision('results_toxicity.csv', '"Neutral"'))
    print(calculate_accuracy('results_toxicity.csv'))
    # print_row('results_toxicity.csv')


if __name__ == "__main__":
    main()
