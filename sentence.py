from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
# Загрузка предобученной модели
model = SentenceTransformer('all-MPNet-base-v2')
import numpy as np
import pickle


# Пример предложений
sentences = [
    "I love programming.",
    "Python is a great programming language.",
    "How to train a machine learning model?"
]

# Получение эмбеддингов
embeddings = model.encode(sentences)

# Вывод эмбеддингов
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding[:5]}...")  # Показать первые 5 элементов эмбеддинга


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Пример текстов
texts = [
    "Machine learning is a subfield of artificial intelligence.",
    "Natural language processing allows computers to understand text.",
    "Convolutional neural networks are effective for image recognition.",
    "AI and ML are transforming industries."
    "Machine learning is a subfield of artificial intelligence.",
    "Natural language processing allows computers to understand text.",
    "Convolutional neural networks are effective for image recognition.",
    "AI and ML are transforming industries."
     "I love programming.",
    "Python is a great programming language.",
    "How to train a machine learning model?"
     "I love programming.",
    "Python is a great programming language.",
    "How to train a machine learning model?"
     "I love programming.",
    "Python is a great programming language.",
    "How to train a machine learning model?"
     "I love programming.",
    "Python is a great programming language.",
    "How to train a machine learning model?",
    "Dual Recurrent Attention Units for Visual Question Answering","We propose an architecture for VQA which utilizes recurrent layers to\ngenerate visual and textual attention. The memory characteristic of the\nproposed recurrent attention units offers a rich joint embedding of visual and\ntextual features and enables the model to reason relations between several\nparts of the image and question. Our single model outperforms the first place\nwinner on the VQA 1.0 dataset, performs within margin to the current\nstate-of-the-art ensemble model. We also experiment with replacing attention\nmechanisms in other state-of-the-art models with our implementation and show\nincreased accuracy. In both cases, our recurrent attention mechanism improves\nperformance in tasks requiring sequential or relational reasoning on the VQA\ndataset.","Ahmed Osman, Wojciech Samek","cs.AI, cs.CL, cs.CV, cs.NE, stat.ML","Dual Recurrent Attention Units for Visual Question Answering. We propose an architecture for VQA which utilizes recurrent layers to\ngenerate visual and textual attention. The memory characteristic of the\nproposed recurrent attention units offers a rich joint embedding of visual and\ntextual features and enables the model to reason relations between several\nparts of the image and question. Our single model outperforms the first place\nwinner on the VQA 1.0 dataset, performs within margin to the current\nstate-of-the-art ensemble model. We also experiment with replacing attention\nmechanisms in other state-of-the-art models with our implementation and show\nincreased accuracy. In both cases, our recurrent attention mechanism improves\nperformance in tasks requiring sequential or relational reasoning on the VQA\ndataset."
    "Sequential Short-Text Classification with Recurrent and Convolutional\n  Neural Networks","Recent approaches based on artificial neural networks (ANNs) have shown\npromising results for short-text classification. However, many short texts\noccur in sequences (e.g., sentences in a document or utterances in a dialog),\nand most existing ANN-based systems do not leverage the preceding short texts\nwhen classifying a subsequent one. In this work, we present a model based on\nrecurrent neural networks and convolutional neural networks that incorporates\nthe preceding short texts. Our model achieves state-of-the-art results on three\ndifferent datasets for dialog act prediction.","Ji Young Lee, Franck Dernoncourt","cs.CL, cs.AI, cs.LG, cs.NE, stat.ML","Sequential Short-Text Classification with Recurrent and Convolutional\n  Neural Networks. Recent approaches based on artificial neural networks (ANNs) have shown\npromising results for short-text classification. However, many short texts\noccur in sequences (e.g., sentences in a document or utterances in a dialog),\nand most existing ANN-based systems do not leverage the preceding short texts\nwhen classifying a subsequent one. In this work, we present a model based on\nrecurrent neural networks and convolutional neural networks that incorporates\nthe preceding short texts. Our model achieves state-of-the-art results on three\ndifferent datasets for dialog act prediction."
    "Multiresolution Recurrent Neural Networks: An Application to Dialogue\n  Response Generation","We introduce the multiresolution recurrent neural network, which extends the\nsequence-to-sequence framework to model natural language generation as two\nparallel discrete stochastic processes: a sequence of high-level coarse tokens,\nand a sequence of natural language tokens. There are many ways to estimate or\nlearn the high-level coarse tokens, but we argue that a simple extraction\nprocedure is sufficient to capture a wealth of high-level discourse semantics.\nSuch procedure allows training the multiresolution recurrent neural network by\nmaximizing the exact joint log-likelihood over both sequences. In contrast to\nthe standard log- likelihood objective w.r.t. natural language tokens (word\nperplexity), optimizing the joint log-likelihood biases the model towards\nmodeling high-level abstractions. We apply the proposed model to the task of\ndialogue response generation in two challenging domains: the Ubuntu technical\nsupport domain, and Twitter conversations. On Ubuntu, the model outperforms\ncompeting approaches by a substantial margin, achieving state-of-the-art\nresults according to both automatic evaluation metrics and a human evaluation\nstudy. On Twitter, the model appears to generate more relevant and on-topic\nresponses according to automatic evaluation metrics. Finally, our experiments\ndemonstrate that the proposed model is more adept at overcoming the sparsity of\nnatural language and is better able to capture long-term structure.","Iulian Vlad Serban, Tim Klinger, Gerald Tesauro, Kartik Talamadupula, Bowen Zhou, Yoshua Bengio, Aaron Courville","cs.CL, cs.AI, cs.LG, cs.NE, stat.ML, I.5.1; I.2.7","Multiresolution Recurrent Neural Networks: An Application to Dialogue\n  Response Generation. We introduce the multiresolution recurrent neural network, which extends the\nsequence-to-sequence framework to model natural language generation as two\nparallel discrete stochastic processes: a sequence of high-level coarse tokens,\nand a sequence of natural language tokens. There are many ways to estimate or\nlearn the high-level coarse tokens, but we argue that a simple extraction\nprocedure is sufficient to capture a wealth of high-level discourse semantics.\nSuch procedure allows training the multiresolution recurrent neural network by\nmaximizing the exact joint log-likelihood over both sequences. In contrast to\nthe standard log- likelihood objective w.r.t. natural language tokens (word\nperplexity), optimizing the joint log-likelihood biases the model towards\nmodeling high-level abstractions. We apply the proposed model to the task of\ndialogue response generation in two challenging domains: the Ubuntu technical\nsupport domain, and Twitter conversations. On Ubuntu, the model outperforms\ncompeting approaches by a substantial margin, achieving state-of-the-art\nresults according to both automatic evaluation metrics and a human evaluation\nstudy. On Twitter, the model appears to generate more relevant and on-topic\nresponses according to automatic evaluation metrics. Finally, our experiments\ndemonstrate that the proposed model is more adept at overcoming the sparsity of\nnatural language and is better able to capture long-term structure."
    "Learning what to share between loosely related tasks","Multi-task learning is motivated by the observation that humans bring to bear\nwhat they know about related problems when solving new ones. Similarly, deep\nneural networks can profit from related tasks by sharing parameters with other\nnetworks. However, humans do not consciously decide to transfer knowledge\nbetween tasks. In Natural Language Processing (NLP), it is hard to predict if\nsharing will lead to improvements, particularly if tasks are only loosely\nrelated. To overcome this, we introduce Sluice Networks, a general framework\nfor multi-task learning where trainable parameters control the amount of\nsharing. Our framework generalizes previous proposals in enabling sharing of\nall combinations of subspaces, layers, and skip connections. We perform\nexperiments on three task pairs, and across seven different domains, using data\nfrom OntoNotes 5.0, and achieve up to 15% average error reductions over common\napproaches to multi-task learning. We show that a) label entropy is predictive\nof gains in sluice networks, confirming findings for hard parameter sharing and\nb) while sluice networks easily fit noise, they are robust across domains in\npractice.","Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, Anders Søgaard","stat.ML, cs.AI, cs.CL, cs.LG, cs.NE","Learning what to share between loosely related tasks. Multi-task learning is motivated by the observation that humans bring to bear\nwhat they know about related problems when solving new ones. Similarly, deep\nneural networks can profit from related tasks by sharing parameters with other\nnetworks. However, humans do not consciously decide to transfer knowledge\nbetween tasks. In Natural Language Processing (NLP), it is hard to predict if\nsharing will lead to improvements, particularly if tasks are only loosely\nrelated. To overcome this, we introduce Sluice Networks, a general framework\nfor multi-task learning where trainable parameters control the amount of\nsharing. Our framework generalizes previous proposals in enabling sharing of\nall combinations of subspaces, layers, and skip connections. We perform\nexperiments on three task pairs, and across seven different domains, using data\nfrom OntoNotes 5.0, and achieve up to 15% average error reductions over common\napproaches to multi-task learning. We show that a) label entropy is predictive\nof gains in sluice networks, confirming findings for hard parameter sharing and\nb) while sluice networks easily fit noise, they are robust across domains in\npractice."
    "A Deep Reinforcement Learning Chatbot","We present MILABOT: a deep reinforcement learning chatbot developed by the\nMontreal Institute for Learning Algorithms (MILA) for the Amazon Alexa Prize\ncompetition. MILABOT is capable of conversing with humans on popular small talk\ntopics through both speech and text. The system consists of an ensemble of\nnatural language generation and retrieval models, including template-based\nmodels, bag-of-words models, sequence-to-sequence neural network and latent\nvariable neural network models. By applying reinforcement learning to\ncrowdsourced data and real-world user interactions, the system has been trained\nto select an appropriate response from the models in its ensemble. The system\nhas been evaluated through AB testing with real-world users, where it\nperformed significantly better than many competing systems. Due to its machine\nlearning architecture, the system is likely to improve with additional data.","Iulian V. Serban, Chinnadhurai Sankar, Mathieu Germain, Saizheng Zhang, Zhouhan Lin, Sandeep Subramanian, Taesup Kim, Michael Pieper, Sarath Chandar, Nan Rosemary Ke, Sai Rajeshwar, Alexandre de Brebisson, Jose M. R. Sotelo, Dendi Suhubdy, Vincent Michalski, Alexandre Nguyen, Joelle Pineau, Yoshua Bengio","cs.CL, cs.AI, cs.LG, cs.NE, stat.ML, I.5.1; I.2.7","A Deep Reinforcement Learning Chatbot. We present MILABOT: a deep reinforcement learning chatbot developed by the\nMontreal Institute for Learning Algorithms (MILA) for the Amazon Alexa Prize\ncompetition. MILABOT is capable of conversing with humans on popular small talk\ntopics through both speech and text. The system consists of an ensemble of\nnatural language generation and retrieval models, including template-based\nmodels, bag-of-words models, sequence-to-sequence neural network and latent\nvariable neural network models. By applying reinforcement learning to\ncrowdsourced data and real-world user interactions, the system has been trained\nto select an appropriate response from the models in its ensemble. The system\nhas been evaluated through AB testing with real-world users, where it\nperformed significantly better than many competing systems. Due to its machine\nlearning architecture, the system is likely to improve with additional data."
    "Generating Sentences by Editing Prototypes","We propose a new generative model of sentences that first samples a prototype\nsentence from the training corpus and then edits it into a new sentence.\nCompared to traditional models that generate from scratch either left-to-right\nor by first sampling a latent sentence vector, our prototype-then-edit model\nimproves perplexity on language modeling and generates higher quality outputs\naccording to human evaluation. Furthermore, the model gives rise to a latent\nedit vector that captures interpretable semantics such as sentence similarity\nand sentence-level analogies.","Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, Percy Liang","cs.CL, cs.AI, cs.LG, cs.NE, stat.ML","Generating Sentences by Editing Prototypes. We propose a new generative model of sentences that first samples a prototype\nsentence from the training corpus and then edits it into a new sentence.\nCompared to traditional models that generate from scratch either left-to-right\nor by first sampling a latent sentence vector, our prototype-then-edit model\nimproves perplexity on language modeling and generates higher quality outputs\naccording to human evaluation. Furthermore, the model gives rise to a latent\nedit vector that captures interpretable semantics such as sentence similarity\nand sentence-level analogies."
    "A Deep Reinforcement Learning Chatbot (Short Version)","We present MILABOT: a deep reinforcement learning chatbot developed by the\nMontreal Institute for Learning Algorithms (MILA) for the Amazon Alexa Prize\ncompetition. MILABOT is capable of conversing with humans on popular small talk\ntopics through both speech and text. The system consists of an ensemble of\nnatural language generation and retrieval models, including neural network and\ntemplate-based models. By applying reinforcement learning to crowdsourced data\nand real-world user interactions, the system has been trained to select an\nappropriate response from the models in its ensemble. The system has been\nevaluated through AB testing with real-world users, where it performed\nsignificantly better than other systems. The results highlight the potential of\ncoupling ensemble systems with deep reinforcement learning as a fruitful path\nfor developing real-world, open-domain conversational agents.","Iulian V. Serban, Chinnadhurai Sankar, Mathieu Germain, Saizheng Zhang, Zhouhan Lin, Sandeep Subramanian, Taesup Kim, Michael Pieper, Sarath Chandar, Nan Rosemary Ke, Sai Rajeswar, Alexandre de Brebisson, Jose M. R. Sotelo, Dendi Suhubdy, Vincent Michalski, Alexandre Nguyen, Joelle Pineau, Yoshua Bengio","cs.CL, cs.AI, cs.LG, cs.NE, stat.ML, I.5.1; I.2.7","A Deep Reinforcement Learning Chatbot (Short Version). We present MILABOT: a deep reinforcement learning chatbot developed by the\nMontreal Institute for Learning Algorithms (MILA) for the Amazon Alexa Prize\ncompetition. MILABOT is capable of conversing with humans on popular small talk\ntopics through both speech and text. The system consists of an ensemble of\nnatural language generation and retrieval models, including neural network and\ntemplate-based models. By applying reinforcement learning to crowdsourced data\nand real-world user interactions, the system has been trained to select an\nappropriate response from the models in its ensemble. The system has been\nevaluated through AB testing with real-world users, where it performed\nsignificantly better than other systems. The results highlight the potential of\ncoupling ensemble systems with deep reinforcement learning as a fruitful path\nfor developing real-world, open-domain conversational agents."
    "Document Image Coding and Clustering for Script Discrimination","The paper introduces a new method for discrimination of documents given in\ndifferent scripts. The document is mapped into a uniformly coded text of\nnumerical values. It is derived from the position of the letters in the text\nline, based on their typographical characteristics. Each code is considered as\na gray level. Accordingly, the coded text determines a 1-D image, on which\ntexture analysis by run-length statistics and local binary pattern is\nperformed. It defines feature vectors representing the script content of the\ndocument. A modified clustering approach employed on document feature vector\ngroups documents written in the same script. Experimentation performed on two\ncustom oriented databases of historical documents in old Cyrillic, angular and\nround Glagolitic as well as Antiqua and Fraktur scripts demonstrates the\nsuperiority of the proposed method with respect to well-known methods in the\nstate-of-the-art.","Darko Brodic, Alessia Amelio, Zoran N. Milivojevic, Milena Jevtic","cs.CV, cs.AI, cs.CL, cs.LG, cs.NE, 97R40, 62H35, 68U15, 68T50,","Document Image Coding and Clustering for Script Discrimination. The paper introduces a new method for discrimination of documents given in\ndifferent scripts. The document is mapped into a uniformly coded text of\nnumerical values. It is derived from the position of the letters in the text\nline, based on their typographical characteristics. Each code is considered as\na gray level. Accordingly, the coded text determines a 1-D image, on which\ntexture analysis by run-length statistics and local binary pattern is\nperformed. It defines feature vectors representing the script content of the\ndocument. A modified clustering approach employed on document feature vector\ngroups documents written in the same script. Experimentation performed on two\ncustom oriented databases of historical documents in old Cyrillic, angular and\nround Glagolitic as well as Antiqua and Fraktur scripts demonstrates the\nsuperiority of the proposed method with respect to well-known methods in the\nstate-of-the-art."
    "Tutorial on Answering Questions about Images with Deep Learning","Together with the development of more accurate methods in Computer Vision and\nNatural Language Understanding, holistic architectures that answer on questions\nabout the content of real-world images have emerged. In this tutorial, we build\na neural-based approach to answer questions about images. We base our tutorial\non two datasets: (mostly on) DAQUAR, and (a bit on) VQA. With small tweaks the\nmodels that we present here can achieve a competitive performance on both\ndatasets, in fact, they are among the best methods that use a combination of\nLSTM with a global, full frame CNN representation of an image. We hope that\nafter reading this tutorial, the reader will be able to use Deep Learning\nframeworks, such as Keras and introduced Kraino, to build various architectures\nthat will lead to a further performance improvement on this challenging task.","Mateusz Malinowski, Mario Fritz","cs.CV, cs.AI, cs.CL, cs.LG, cs.NE","Tutorial on Answering Questions about Images with Deep Learning. Together with the development of more accurate methods in Computer Vision and\nNatural Language Understanding, holistic architectures that answer on questions\nabout the content of real-world images have emerged. In this tutorial, we build\na neural-based approach to answer questions about images. We base our tutorial\non two datasets: (mostly on) DAQUAR, and (a bit on) VQA. With small tweaks the\nmodels that we present here can achieve a competitive performance on both\ndatasets, in fact, they are among the best methods that use a combination of\nLSTM with a global, full frame CNN representation of an image. We hope that\nafter reading this tutorial, the reader will be able to use Deep Learning\nframeworks, such as Keras and introduced Kraino, to build various architectures\nthat will lead to a further performance improvement on this challenging task."
    "pix2code: Generating Code from a Graphical User Interface Screenshot","Transforming a graphical user interface screenshot created by a designer into\ncomputer code is a typical task conducted by a developer in order to build\ncustomized software, websites, and mobile applications. In this paper, we show\nthat deep learning methods can be leveraged to train a model end-to-end to\nautomatically generate code from a single input image with over 77% of accuracy\nfor three different platforms (i.e. iOS, Android and web-based technologies).","Tony Beltramelli","cs.LG, cs.AI, cs.CL, cs.CV, cs.NE, 68T45, I.2.1; I.2.10; I.2.2; I.2.6","pix2code: Generating Code from a Graphical User Interface Screenshot. Transforming a graphical user interface screenshot created by a designer into\ncomputer code is a typical task conducted by a developer in order to build\ncustomized software, websites, and mobile applications. In this paper, we show\nthat deep learning methods can be leveraged to train a model end-to-end to\nautomatically generate code from a single input image with over 77% of accuracy\nfor three different platforms (i.e. iOS, Android and web-based technologies)."

]

# Инициализация кастомной модели эмбеддинга (можно заменить на любую Sentence-BERT модель)
embedding_model = SentenceTransformer('all-MPNet-base-v2')

from umap import UMAP
from hdbscan import HDBSCAN

# Создание кастомного кластеризатора
hdbscan_model = HDBSCAN(min_samples=5, min_cluster_size=5)
from sklearn.feature_extraction.text import TfidfVectorizer

# Используем KeyBERTInspired для извлечения ключевых слов
representation_model = KeyBERTInspired()


vectorizer_model = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')

umap_model = UMAP(n_neighbors=5, n_components=2, random_state=42)
topic_model = BERTopic(
    representation_model=representation_model,
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model,
    umap_model=umap_model
)
# Обучение модели
topics, probs = topic_model.fit_transform(texts)
# Получение информации о темах
topics_info = topic_model.get_topic_info()
print("Информация о темах:")
print(topics_info)

print(topics_info.head())
print(topics_info.columns)
print(topics_info[topics_info['Topic'] != -1])

# Печать ключевых слов для шумовой темы (-1)
print("Ключевые слова для шумовой темы (-1):")
print(topic_model.get_topic(-1,))  # Шумовая тема
# Получение ключевых слов для каждой темы
for topic_id in topics_info['Topic']:
    if topic_id != -1:  # Игнорируем "шумовые" данные
        print(f"\nКлючевые слова для темы {topic_id}:")
        print(topic_model.get_topic(topic_id))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Уменьшение размерности для визуализации
embeddings = embedding_model.encode(texts)
print('SDFDSFDSFDSFDSFDS')
print('1',embeddings)




pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Визуализация
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=probs)
plt.colorbar()
plt.show()
print(topic_model.get_topics())  # Проверка топиков
print(topic_model.get_topic_info())  # Общая информация о темах

topic_model.visualize_topics().write_html("topic_visualization1.html")
'''
import json

# Сохраняем текст и его эмбеддинг как пары
embeddings_data = [{"text": text, "embedding": emb.tolist()} for text, emb in zip(texts, embeddings)]

# Записываем в JSON
with open("embeddings_with_texts.json", "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=4)

'''