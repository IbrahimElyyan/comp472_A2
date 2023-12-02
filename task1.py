import csv
import json
import gensim.downloader

# Load the pre-trained Word2Vec model
model = gensim.downloader.load('word2vec-google-news-300')

# Load the Synonym Test dataset
with open('synonym.json', 'r') as f:
    data = json.load(f)

# Prepare the output files
details_file = open('word2vec-google-news-300-details.csv', 'w', newline='')
analysis_file = open('analysis.csv', 'w', newline='')
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
analysis_writer.writerow(['word2vec-google-news-300', len(model.key_to_index), correct_count, total_count, correct_count / total_count])

# Close the output files
details_file.close()
analysis_file.close()
