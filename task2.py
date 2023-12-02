import csv
import json
import gensim.downloader

# Define the models to use
models = [
    'word2vec-google-news-300',
    'glove-twitter-25',
    'glove-twitter-100',
    'glove-wiki-gigaword-50',
    'glove-wiki-gigaword-100'
]

# Load the Synonym Test dataset
with open('synonym.json', 'r') as f:
    data = json.load(f)

# Iterate over the models
for model_name in models:
    # Load the pre-trained Word2Vec model
    model = gensim.downloader.load(model_name)

    # Prepare the output files
    details_file = open(f'{model_name}-details.csv', 'w', newline='')
    analysis_file = open('analysis.csv', 'a', newline='')  # Append to the existing file
    details_writer = csv.writer(details_file)
    analysis_writer = csv.writer(analysis_file)

    correct_count = 0
    total_count = 0

    # Iterate over the dataset
    for item in data:
        question_word = item['question']
        guess_words = item['choices']
        correct_word = item['answer']

        if question_word in model:
            similarities = [(word, model.similarity(question_word, word)) for word in guess_words if word in model]
            if similarities:
                total_count += 1
                guess_word, _ = max(similarities, key=lambda x: x[1])
                if guess_word == correct_word:
                    correct_count += 1
                    details_writer.writerow([question_word, correct_word, guess_word, 'correct'])
                else:
                    details_writer.writerow([question_word, correct_word, guess_word, 'wrong'])
            else:
                details_writer.writerow([question_word, correct_word, '', 'guess'])
        else:
            details_writer.writerow([question_word, correct_word, '', 'guess'])

    # Write the analysis
    analysis_writer.writerow([model_name, len(model.key_to_index), correct_count, total_count, correct_count / total_count if total_count > 0 else 0])

    # Close the output files
    details_file.close()

# Close the analysis file
analysis_file.close()
