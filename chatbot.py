import random
from nltk.metrics.distance import edit_distance

from dataset import BOT_CONFIG

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

CLASSIFIER_THRESHOLD = 0.3
GENERATIVE_THRESHOLD = 0.7

# Dataset preparation
X_texts = []
y = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_texts.append(example)
        y.append(intent)

# TODO: Можно удалить стоп слова
STOPWORDS = ['я', 'у', 'и', 'а', 'в', 'к', 'с', 'да', 'на', '...', 'бы', 'о']

# Text Vectorization
vectorizer = TfidfVectorizer(analyzer='char_wb', stop_words=STOPWORDS,
                             norm='l2', ngram_range=(2, 4))
X = vectorizer.fit_transform(X_texts)
clf = LinearSVC().fit(X, y)
clf2 = SVC(probability=True).fit(X, y)

def get_intent(text):
    question_vector = vectorizer.transform([text])
    intent = clf.predict(vectorizer.transform([text]))[0]

    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        dist = edit_distance(filter_text(text), example)
        dist_percentage = dist / len(example)
        if dist_percentage < CLASSIFIER_THRESHOLD:
            return intent

    # my_index = list(clf2.classes_).index(intent)
    # proba = clf2.predict_proba(question_vector)[0][my_index]
    # print(intent, proba)
    # if proba > CLASSIFIER_THRESHOLD:
    # return intent


rus_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -'
ger_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜß'.lower()
# numbers = '0123456789'
# symbols = '!@#$%&? -'
alphabet = rus_alphabet + ger_alphabet  # + numbers + symbols


def filter_text(text):
    text = text.lower()
    text = [c for c in text if c in alphabet]
    text = ''.join(text)
    return text


# Dialogues text preparation
# Dialogue data comes from
# https://github.com/Koziev/NLP_Datasets/blob/master/Conversations/Data/dialogues.zip

with open('dialogues.txt') as f:
    content = f.read()

dialogues = [dialogue_line.split('\n') for dialogue_line in content.split('\n\n')]

qa_dataset = []
questions = set()

for replicas in dialogues:
    if len(replicas) < 2:
        continue

    question, answer = replicas[:2]
    answer = answer[2:]
    question = filter_text(question[2:])

    if question and question not in questions:
        questions.add(question)
        qa_dataset.append([question, answer])

qa_by_word_dataset = {}
for question, answer in qa_dataset:
    words = question.split(' ')
    for word in words:
        if word not in qa_by_word_dataset:
            qa_by_word_dataset[word] = []
        qa_by_word_dataset[word].append((question, answer))

qa_by_word_dataset_filtered = {word: qa_list
                               for word, qa_list in qa_by_word_dataset.items()
                               if len(qa_list) < 2500}


def generate_answer(text):
    text = filter_text(text)
    words = text.split(' ')
    qa = []
    for word in words:
        if word in qa_by_word_dataset_filtered:
            qa += qa_by_word_dataset_filtered[word]
    qa = list(set(qa))

    results = []
    for question, answer in qa:
        distance = edit_distance(question, text)
        dist_percentage = distance / len(question)
        results.append([distance / len(question), question, answer])

    if results:
        dist, question, answer = min(results, key=lambda pair: pair[0])
        if dist < GENERATIVE_THRESHOLD:
            return answer


def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


def get_default_response():
    candidates = BOT_CONFIG['failure_phrases']
    return random.choice(candidates)


stats = [0, 0, 0]


def get_answer(text):
    # NLU
    intent = get_intent(text)
    if intent:
        response = get_response_by_intent(intent)
        if response:
            stats[0] += 1
            return get_response_by_intent(intent)

    # generative model
    response = generate_answer(text)
    if response:
        stats[1] += 1
        return response

    # default answer
    stats[2] += 1
    return get_default_response()


question = None

# Telegram functions come from
# https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/echobot.py

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    answer = get_answer(update.message.text)
    update.message.reply_text(answer)


def main():
    """Start the bot."""
    updater = Updater("TOKEN", use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()
    updater.idle()


main()
