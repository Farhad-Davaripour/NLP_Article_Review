h Introduction
Natural Language Processing (`NLP`) is a field of Artificial Intelligence (`AI`) that deals with the interaction between human language and computers. Some real life applications of NLP are: Google Translate, Chat bots, Grammarly, Chat GPT, and etc [3]. 

# Essential Steps in a Text Processing Pipeline
The common steps in Natural Language Processing (NLP) involves:  

1\. `Tokenization`: The process of breaking down sentences into individual word also called tokens. This is typically the first step in converting unstructured data into meaningful information [1]. Example:  
><pre> Sentence = "John decided to go to New York to attend the conference."</pre>
><pre> Tokens: ['John', 'decided', 'to', 'go', 'to', 'New', 'York', 'to', 'attend', 'the', 'conference', '.']</pre>
2\. `Stop Words`: This step eliminates the words that do not contribute to the overall meaning of the sentence also called `stop words`. Using above example [1]:  
><pre> Filtered Sentence: ['John', 'decided', 'go', 'New', 'York', 'attend', 'conference', '.']</pre>
3\. `Word Normalization`: It includes any techniques that reduces the word into it's base or root form. Two techniques employed in word normalization are [1]:
- `Stemming`: This step converts the words into their root form. Using above example:  
><pre> Stemmed words: ['john', 'decid', 'go', 'new', 'york', 'attend', 'confer', '.']></pre>
-  `Lemmatization`: Similar to stemming but also takes into account the context of the word which makes it computationally more expensive. The lemmatization is typically a preferred normalization technique due to it's comprehensive morphological analyses. Using above example:  
><pre> Lemmatized words: ['John', 'decided', 'go', 'New', 'York', 'attend', 'conference', '.']</pre>
4\. `Part of Speech (POSE)`: This technique classifies the words into the grammatical categories (e.g., verb, noun, etc.) [1]. Using above example:  
><pre> POS tags: [('John', 'NNP'), ('decided', 'VBD'), ('go', 'VB'), ('New', 'NNP'), ('York', 'NNP'), ('attend', 'JJ'), ('conference', 'NN'), ('.', '.')]</pre>
5\. `Named Entity Recognition (NER)`: This method classifies words into predefined categories such as person, organization, etc [1]. Using above example:  
><pre> Named entities: (S(PERSON John/NNP) decided/VBD to/TO go/VB to/TO (GPE New/NNP York/NNP) to/TO attend/VB the/DT conference/NN ./.) </pre>
6\. `Bag of Words (BoW)`: This is a popular technique for feature extraction which counts the frequency of each unique word within a document. One issue with BoW is that when the vocabulary (i.e., set of unique words across all documents) is large and a given document contains only a small subset of this vocabulary, the BoW representation for this document will be filled with zeros for all the absent words. This leads to a sparse vector representation, which can be computationally challenging to handle due to its size and the fact that it mostly contains irrelevant (zero) information [2].    

7\. `N-gram`: It provides a more detailed understanding of context compared to BoW by providing identifying the sequence of n consecutive and their frequency [2]. Example:
> Text: 'The quick brown fox jumped over the lazy dog.'
><pre>Bigrams: [('The', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumped'), ('jumped', 'over'), ('over', 'the'), ('the', 'lazy'), ('lazy', 'dog'), ('dog', '.')]</pre>
><pre>Trigrams: [('The', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('brown', 'fox', 'jumped'), ('fox', 'jumped', 'over'), ('jumped', 'over', 'the'), ('over', 'the', 'lazy'), ('the', 'lazy', 'dog'), ('lazy', 'dog', '.')]</pre>
8\. `Term Frequency-Inverse Document Frequency (TF-IDF)`: It is a statistical measure to evaluate the importance of a word in a document relative to it's frequency in all documents. It aims at finding a balance between local relevance (how often a word appears in a specific document) and global rarity (how uncommon the word is across the entire collection of documents). [2]. The first part (`TF`) determines the frequency of the word within the document and the second part (`IDF`) inversely weighs the frequency of the words within the whole corpus. The TF-IDF score is then the multiplication of these two factors. Note that tokenizing, removing punctuations, lower casing the words are the steps taken prior to performing TF-IDF:
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

9\. `Word Embedding`: It is a technique to convert the words within a document to a multi dimensional vector space. Word2Vec is one popular method used in word embedding, which employs two techniques: Continuous Bag of Words (`CBOW`) and `Skip Gram`, where the former predicts the target word based on the surrounding words, while the latter predicts the surrounding words based on the target word. The two technique are then used in Word2Vec to generate word embeddings and capture the semantic relationship between the words.  

The information that could be represented by fixed_length vectors (i.e., the number of elements within each vector is constant) include but not limited to: 
- semantic meaning: words with similar meaning are closer to each other in the vector space.
- Syntactic information: The vectors could holds information on the word's part of the speech.
- Morphological information: This includes things like the verb tense or the plurality of the nouns that could be represented by vectors.
- Contextual usage: Words that appears in similar context (sport context) tend to form a cluster within the vector space.

As an example, the sentences of the first 20% of the `Brown University Standard Corpus of Present-Day American English` are vectorized using Word2Vec. In order to visualize it, Principal Component Analysis (PCA) is used to reduce the vector space to 2 dimensions. The plot below shows the vectorized tokens with respect to PC1 and PC2. These components represent directions in which the data varies the most, and positioning of tokens in this space can give insights into their relationships in terms of semantic meaning, although the specific interpretation can be complex.   

 <img src="Figures\PCA_Word2Vec.png" alt="Word2Vec_Biplot" width="600"/>     

 Check out the code snippet which generated this plot from the link below:
 [Link](https://colab.research.google.com/drive/1KzpbhtlThgvFZf5UZNL1vnQ5de13lM9W)    

10\. `Language Models (LM)`: LM plays a fundamental role in NLP tasks primarily used to predict the next word in the sequence based on the words already observed. The `Recurrent Neural Network (RNN)` has been critical in the development of modern LMs, as unlike traditional neural networks, it contains loops that make it particularly suitable for tasks that involve sequence of data like text [4]. This is done by including a hidden cell which function as a memory system. These cells are continuously updated and hold a compressed, fixed-length vector representation of the previous sequence, which is utilized to predict the next word in the sequence.    

11\. `Contextual Word EMbedding`: Unlike the traditional word embedding which assigns one single meaning to a unique word or in other words represents each unique word with a corresponding unique vector, in contextual word embedding, a unique word could have multiple vector representations based on the context it was used in in the document. This is done using LMs to understand the meaning of the word in a given context. This helps to handle embeddings for words that could have a totally difference meaning when used in a difference context (e.g., spring). ELMo which is short for `Embeddings from Language Models` is used for contextual work embeddings which utilizes multi layer bidirectional Long Short Term Memory (`LSTM`) models which is a variation of the RNN architecture.   

12\. `Transfer Learning`: This concept involves leveraging a pre-trained model for a new but related task. Within the domain of transfer learning, methods like `Discriminative Fine Tuning` and `Slanted Triangular Learning` are grouped under `Universal Language Fine Tuning (ULfit)`. Discriminative Fine Tuning adjusts different layers at varying (learning) rates as part of the fine-tuning process, while Slanted Triangular Learning first adapts quickly, then more slowly. This approach obtains a balance by quickly adapting to the new task while at the same time retaining the knowledge gained during the pre-training process [4].   

13\. Sentence Embedding: Similar to word embedding but take it one step further and vectorize sentences, for instance simply by averaging the vector of each word within the sentence. There are also more sophisticated methods such as `Skip Thought Vector` or `Quick Thought Vector`to maintain more information such as the relationship between the words [4].

14\. `Sequence to Sequence (Seq2Seq)`: It is a class of models typically used to convert sentences from one domain to another (e.g., translation from one language to another). It could also be used in tasks like text summarization. It encompasses two main components: `encoder` and `decoder`, where the former compresses the essence of the input text in a form of vectors, and the latter generate outputs in for instance target language based on the input context vectors.

15\. `Attention Mechanism`: This is an improvement to the decoding component in Seq2Seq models which allows better handling of long sequence. In traditional Seq2Seq, the encoder final hidden state (which is a single fixed length vector) includes the entire input sequence which could cause issues specially in long documents. The attention mechanism addresses this by allowing the decoder to look at different parts of the sentence (i.e., giving different parts different weights) as opposed to only looking at the final hidden state.

16\. `Transformers`: The transformer deals with sequence data, but unlike RNN it incorporates attention mechanism which considerably enhances its robustness. In simple terms, instead of using a single fixed-length vector like RNN, the transformer uses all the vectorized tokens within the document (that are processed by the attention mechanism based on e.g., context and similarity) simultaneously.

17\. `GPT and BERT`: OpenAI's GPT and Google's BERT utilize transfer learning and attention mechanisms in NLP. Unlike LSTM, they process different parts of sentences. BERT's distinct feature is its bidirectional approach, examining both sides of a word during training, along with a "masking" technique to prevent future information from influencing current training. This contrasts with GPT's unidirectional focus.

18\. `GPT2, GPT3, and Extra Large Language Model (XLNT)`: OpenAI's GPT-2 and GPT-3 are the improved versions of the original Generative Pre-trained Transformer (GPT) by OpenAI. The original GPT model had 117 million parameters, GPT-2 expanded this to 1.5 billion parameters, and GPT-3 further increased the size to an astonishing 175 billion parameters.

Google, in collaboration with Carnegie Mellon University, developed the Extra Large Language Model (XLNet). Unlike BERT, which utilizes a `bidirectional transformer` and processes all words in the input sequence simultaneously, XLNet employs an `autoregressive approach` that predicts each word sequentially, incorporating all previous words in the process. This allows XLNet to capture the dependencies between the words in a sequence more effectively than BERT's bidirectional method. By considering the entire context of a given word rather than just the preceding or subsequent words, XLNet has been able to outperform BERT in several tasks. 

19\. `Sentiment Analysis`: Sentiment analysis is a computational technique used to detect and categorize attitudes and opinions in text, applicable at various granularity levels like documents, sentences, or paragraphs. 

Generally there are three approached to perform sentiment analysis. When dealing with specialized industry-specific language, it might be necessary to fine-tune a large language model to ensure it understands the unique terminology. If the document does not contain industry-specific language, pre-trained large-scale models (LLMs) can be used directly. Though training a model from scratch is an option, it requires considerable effort and resources, so it's generally reserved as a last resort when absolutely necessary.

20\. `Multimodel learning`: Multimodal learning is an innovative approach that seeks to emulate the way the human brain processes information by integrating data from different sources like text, images, and audio. This integration can significantly enhance various applications, such as providing captions to images based on text, or searching using both visual and text data.

The encoding and vectorizing of different data sources are achieved through:
Text: Conversion into numerical values using embeddings like Word2Vec, developed by Google.
Images: Feature extraction and vectorization through Convolutional Neural Networks (CNNs).
Audio: Utilization of spectrograms or time-frequency domain techniques for feature extraction and vectorization.

Once vectorized, these diverse data sources are integrated using various fusion techniques:
Early Fusion: A simple concatenation at the early stage of processing.
Intermediate Fusion: Utilizes attention mechanisms to weight data from different sources based on their relevance or contribution.
Late Fusion: Training separate models for each data type and combining their predictions at the end.

21\. `Question Answering (QA)`: Question and Answering (QA) systems are diverse and sophisticated, encompassing different categories and approaches. `Factoid QA` provides straightforward answers to specific queries such as "Who" or "When," while `Non-Factoid QA` delves into more complex interpretations and reasoning for questions like "Why is the sky blue?" `Visual QA` answers questions about images or videos, and the field also includes `Open-Domain QA`, covering general knowledge, and `Closed-Domain QA`, which focuses on specialized areas like medicine or engineering. Various strategies can be employed to answer these questions, ranging from `Rule-Based methods` that use predefined rules, `Retrieval-Based methods` that fetch relevant content, `Generative approaches` that train models to answer, to `Hybrid systems` that combine different techniques. Modern technology's advancement has been marked by the utilization of large language models like BERT or GPT, enhancing the capability and accuracy of QA systems across these diverse domains.

22\. `Information Retrieval (IR)`: IR is a fundamental component of search engines like Google. It involves collecting and retrieving relevant information based on user queries from vast databases filled with text data. Some of the core technologies in this field include:
Boolean Model: This approach searches for specific tokens mentioned in a query and retrieves documents containing all of them, such as "apple" and "banana."
Vector-Based Models: These models represent documents and queries as vectors in multi-dimensional space. The system finds the documents with the smallest cosine distance to the query vector, effectively locating the most relevant documents.
In addition to these technologies, there are other significant areas in IR:
Personalization: This tailors search results according to the user's behavior, providing more individualized results.
Page Ranking: Algorithms like PageRank assess and rank pages based on the quality and quantity of links pointing to a particular document or website. This ensures the most authoritative sources are displayed prominently.
Indexing: By organizing data into a structured format, indexing allows faster searches, making the retrieval process more efficient.
Query Expansion: This technique involves expanding a query by finding synonyms or related terms. By broadening the search, it helps to find more relevant documents.

# Libraries
The following libraries are quite powerful and powerful in performing NLP tasks:
- Natural Language Toolkit (`NLTK`): It is a python library which encompasses text processing libraries to perform tasks such as: Tokenization, Classification, Stemming, Tagging, Parsing, etc. 
- Beautiful Soup 4 (`BS4`): It is a python library used to pull specific content from HTML and XML documents, assisting to cleanup the markups and HTML tags.

# Reference:
1. [Towards AI: Natural Language Processing Beginner’s Guide](https://pub.towardsai.net/natural-language-processing-beginners-guide-461d569b70e3)
2. [Medium: Natural Language Processing Basics for Beginners](https://medium.com/p/ba1577b7a5f8)
3. [Towards Data Science: Natural Language Processing (NLP) for Beginners](https://towardsdatascience.com/natural-language-processing-nlp-for-beginners-6d19f3eedfea)
4. [Towards Data Science: A collection of must known resources for every Natural Language Processing (NLP) practitioner](https://towardsdatascience.com/a-collection-of-must-known-pre-requisite-resources-for-every-natural-language-processing-nlp-a18df7e2e027)

