# Introduction
Natural Language Processing (`NLP`) is a field of Artificial Intelligence (`AI`) that deals with the interaction between human language and computers. Some real life applications of NLP are: Google Translate, Chat bots, Grammarly, Chat GPT, and etc [3]. 

# Steps
The common steps in Natural Language Processing (NLP) involves:
1. `Tokenization`: The process of breaking down sentences into individual word also called tokens. This is typically the first step in converting unstructured data into meaningful information [1]. Example:  
><pre> Sentence = "John decided to go to New York to attend the conference."</pre>
><pre> Tokens: ['John', 'decided', 'to', 'go', 'to', 'New', 'York', 'to', 'attend', 'the', 'conference', '.']</pre>
2. `Stop Words`: This step eliminates the words that do not contribute to the overall meaning of the sentence also called `stop words`. Using above example [1]:  
><pre> Filtered Sentence: ['John', 'decided', 'go', 'New', 'York', 'attend', 'conference', '.']</pre>
3. `Word Normalization`: It includes any techniques that reduces the word into it's base or root form. Two techniques employed in word normalization are [1]:
- `Stemming`: This step converts the words into their root form. Using above example:  
><pre> Stemmed words: ['john', 'decid', 'go', 'new', 'york', 'attend', 'confer', '.']></pre>
-  `Lemmatization`: Similar to stemming but also takes into account the context of the word which makes it computationally more expensive. The lemmatization is typically a preferred normalization technique due to it's comprehensive morphological analyses. Using above example:  
><pre> Lemmatized words: ['John', 'decided', 'go', 'New', 'York', 'attend', 'conference', '.']</pre>
4. `Part of Speech (POSE)`: This technique classifies the words into the grammatical categories (e.g., verb, noun, etc.) [1]. Using above example:  
><pre> POS tags: [('John', 'NNP'), ('decided', 'VBD'), ('go', 'VB'), ('New', 'NNP'), ('York', 'NNP'), ('attend', 'JJ'), ('conference', 'NN'), ('.', '.')]</pre>
5. `Named Entity Recognition (NER)`: This method classifies words into predefined categories such as person, organization, etc [1]. Using above example:  
><pre> Named entities: (S(PERSON John/NNP) decided/VBD to/TO go/VB to/TO (GPE New/NNP York/NNP) to/TO attend/VB the/DT conference/NN ./.) </pre>
6. `Bag of Words (BoW)`: This is a popular technique for feature extraction which counts the frequency of each unique word within a document. One issue with BoW is that when the vocabulary (i.e., set of unique words across all documents) is large and a given document contains only a small subset of this vocabulary, the BoW representation for this document will be filled with zeros for all the absent words. This leads to a sparse vector representation, which can be computationally challenging to handle due to its size and the fact that it mostly contains irrelevant (zero) information [2].
7. `N-gram`: It provides a more detailed understanding of context compared to BoW by providing identifying the sequence of n consecutive and their frequency [2]. Example:
> Text: 'The quick brown fox jumped over the lazy dog.'
><pre>Bigrams: [('The', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumped'), ('jumped', 'over'), ('over', 'the'), ('the', 'lazy'), ('lazy', 'dog'), ('dog', '.')]</pre>
><pre>Trigrams: [('The', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('brown', 'fox', 'jumped'), ('fox', 'jumped', 'over'), ('jumped', 'over', 'the'), ('over', 'the', 'lazy'), ('the', 'lazy', 'dog'), ('lazy', 'dog', '.')]</pre>
8. `Term Frequency-Inverse Document Frequency (TF-IDF)`: It is a statistical measure to evaluate the importance of a word in a document relative to it's frequency in all documents. It aims at finding a balance between local relevance (how often a word appears in a specific document) and global rarity (how uncommon the word is across the entire collection of documents). [2]. The first part (`TF`) determines the frequency of the word within the document and the second part (`IDF`) inversely weighs the frequency of the words within the whole corpus. The TF-IDF score is then the multiplication of these two factors. Note that tokenizing, removing punctuations, lower casing the words are the steps taken prior to performing TF-IDF:
> Documents:     
    "John likes to watch movies, especially horror movies.",  
    "Mary likes movies too.",  
    "John also likes to watch football games."  
  
|     | also     | especially | football | games    | horror   | john     | likes    | mary     | movies   | to       | too      | watch    |
|-----|----------|------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| doc #1| 0.000000 | 0.395358   | 0.000000 | 0.000000 | 0.395358 | 0.300680 | 0.233505 | 0.000000 | 0.601360 | 0.300680 | 0.000000 | 0.300680 |
| doc #2| 0.000000 | 0.000000   | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.345205 | 0.584483 | 0.444514 | 0.000000 | 0.584483 | 0.000000 |
| doc #3| 0.443503 | 0.000000   | 0.443503 | 0.443503 | 0.000000 | 0.337295 | 0.261940 | 0.000000 | 0.000000 | 0.337295 | 0.000000 | 0.337295 |

The words with higher score indicates that the word is specifics to that document and does not appear in others and hence more informative. 

9. `Word Embedding`: It is a technique to convert the words within a document to a multi dimensional vector space. Word2Vec is one popular method used in word embedding, which employs two techniques: Continuous Bag of Words (`CBOW`) and `Skip Gram`, where the former predicts the target word based on the surrounding word, while the latter predicts the surrounding words based on the target word. The two technique are then used in Word2Vec to generate word embeddings and capture the semantic relationship between the words.
 
# Libraries
The following libraries are quite powerful and powerful in performing NLP tasks:
- Natural Language Toolkit (`NLTK`): It is a python library which encompasses text processing libraries to perform tasks such as: Tokenization, Classification, Stemming, Tagging, Parsing, etc. 
- Beautiful Soup 4 (`BS4`): It is a python library used to pull specific content from HTML and XML documents, assisting to cleanup the markups and HTML tags.

# Reference:
1. [Towards AI: Natural Language Processing Beginner’s Guide](https://pub.towardsai.net/natural-language-processing-beginners-guide-461d569b70e3)
2. [Medium: Natural Language Processing Basics for Beginners](https://medium.com/p/ba1577b7a5f8)
3. [Towards Data Science: Natural Language Processing (NLP) for Beginners](https://towardsdatascience.com/natural-language-processing-nlp-for-beginners-6d19f3eedfea)