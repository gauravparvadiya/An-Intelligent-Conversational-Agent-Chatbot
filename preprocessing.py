import re

def load_data():
    lines = open('data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    return lines, conv_lines

def generate_question_answer_from_data(lines,conv_lines):
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))

    questions = []
    answers = []

    for conv in convs:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])

    return questions,answers

def clean_data(text):
    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

def clean_questions(questions):
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_data(question))

    return clean_questions

def clean_answers(answers):
    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_data(answer))

    return clean_answers

def filter_long_short_questions_answers(questions, answers, min_length, max_length):
    # Remove questions and answers that are shorter than 2 words and longer than 20 words.
    min_line_length = min_length
    max_line_length = max_length

    # Filter out the questions that are too short/long
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in questions:
        if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
            short_questions_temp.append(question)
            short_answers_temp.append(answers[i])
        i += 1

    # Filter out the answers that are too short/long
    short_questions = []
    short_answers = []

    i = 0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
            short_answers.append(answer)
            short_questions.append(short_questions_temp[i])
        i += 1

    return short_questions,short_answers

def question_answer_vocab_to_int(questions, answers, threshold):
    vocab = {}
    for question in questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    questions_vocab_to_int = {}

    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            questions_vocab_to_int[word] = word_num
            word_num += 1

    answers_vocab_to_int = {}

    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            answers_vocab_to_int[word] = word_num
            word_num += 1

    return questions_vocab_to_int, answers_vocab_to_int

def add_tokens(questions_vocab_to_int,answers_vocab_to_int,short_questions,short_answers):
    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

    for code in codes:
        questions_vocab_to_int[code] = len(questions_vocab_to_int) + 1

    for code in codes:
        answers_vocab_to_int[code] = len(answers_vocab_to_int) + 1

    for i in range(len(short_answers)):
        short_answers[i] += ' <EOS>'

    return questions_vocab_to_int,answers_vocab_to_int,short_questions,short_answers

def convert_to_int(questions_vocab_to_int,answers_vocab_to_int,short_questions,short_answers):
    questions_int = []
    for question in short_questions:
        ints = []
        for word in question.split():
            if word not in questions_vocab_to_int:
                ints.append(questions_vocab_to_int['<UNK>'])
            else:
                ints.append(questions_vocab_to_int[word])
        questions_int.append(ints)

    answers_int = []
    for answer in short_answers:
        ints = []
        for word in answer.split():
            if word not in answers_vocab_to_int:
                ints.append(answers_vocab_to_int['<UNK>'])
            else:
                ints.append(answers_vocab_to_int[word])
        answers_int.append(ints)

    return questions_int,answers_int

def sort_question_answer(questions,answers,max_length):
    sorted_questions = []
    sorted_answers = []

    for length in range(1, max_length + 1):
        for i in enumerate(questions):
            if len(i[1]) == length:
                sorted_questions.append(questions[i[0]])
                sorted_answers.append(answers[i[0]])

    return sorted_questions,sorted_answers