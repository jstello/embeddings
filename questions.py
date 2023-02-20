import os
import re
import openai
import argparse

openai.api_key = "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"

parser = argparse.ArgumentParser(description='Answer traffic law questions')
parser.add_argument('question', type=str, help='traffic law question')
args = parser.parse_args()

corpus_path = 'PATH_TO_CORPUS_FOLDER'

sentences = []

for filename in os.listdir(corpus_path):
    if filename.endswith('.txt'):
        with open(os.path.join(corpus_path, filename), encoding='utf-8') as file:
            corpus = file.read()
            corpus_sentences = re.split("(?<=[.!?]) +", corpus)
            for sentence in corpus_sentences:
                sentences.append(sentence)

embeddings = []
for sentence in sentences:
    response = openai.Embedding.create(
        input=sentence,
        model="text-embedding-ada-002"
    )
    embeddings.append(response['data'][0]['embedding'])

question = args.question
question_embedding = openai.Embedding.create(
    input=question,
    model="text-embedding-ada-002"
)['data'][0]['embedding']

similarities = [float(openai.Tools.cos_sim(question_embedding, embedding)) for embedding in embeddings]

most_similar = max(similarities)
index = similarities.index(most_similar)

print(sentences[index])
