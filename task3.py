import csv
import json
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

# Define the parameters to use
parameters = [
    {'embedding_size': 100, 'window_size': 5},
    {'embedding_size': 100, 'window_size': 10},
    {'embedding_size': 200, 'window_size': 5},
    {'embedding_size': 200, 'window_size': 10}
]

# Load the Synonym Test dataset
with open('synonym.json', 'r') as f:
    data = json.load(f)

# Preprocess the corpus
corpus = []
for book_filename in ['book1.txt', 'book2.txt', 'book3.txt', 'book4.txt', 'book5.txt']: 
    with open(book_filename, 'r', encoding='utf-8') as f:
        book_text = f.read()
        sentences = sent_tokenize(book_text)
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        corpus.extend(tokenized_sentences)


# Iterate over the parameters
for params in parameters:
    # Train the Word2Vec model
    model = Word2Vec(sentences=corpus, vector_size=params['embedding_size'], window=params['window_size'])

    # Prepare the output files
    model_name = f'word2vec-gutenberg-{params["embedding_size"]}-{params["window_size"]}'
    details_file = open(f'{model_name}-details.csv', 'w', newline='')
    analysis_file = open('analysis.csv', 'a', newline='')  # Append to the existing file
    details_writer = csv.writer(details_file)
    analysis_writer = csv.writer(analysis_file)

    c = 0
    v = 0

    # Iterate over the dataset
    for item in data:
        question_word = item['question']
        guess_words = item['choices']
        correct_word = item['answer']

        if question_word in model.wv:
            similarities = [(word, model.wv.similarity(question_word, word)) for word in guess_words if word in model.wv]
            if similarities:
                v += 1
                guess_word, _ = max(similarities, key=lambda x: x[1])
                if guess_word == correct_word:
                    c += 1
                    details_writer.writerow([question_word, correct_word, guess_word, 'correct'])
                else:
                    details_writer.writerow([question_word, correct_word, guess_word, 'wrong'])
            else:
                details_writer.writerow([question_word, correct_word, '', 'guess'])
        else:
            details_writer.writerow([question_word, correct_word, '', 'guess'])

    # Write the analysis
    analysis_writer.writerow([model_name, len(model.wv.key_to_index), c, v, c / v if v > 0 else 0])

    # Close the output files
    details_file.close()

# Close the analysis file
analysis_file.close()
