Predict your CEFR level by answering a random prompt

**Model trained on 10 features:**

- Number of sentences
- Average sentence length
- Standard deviation of the sentence lengths
- Average character count per word
- Content word proportion (e.g. nouns, verbs, adjectives, adverbs)
- Count of derivational morphology (e.g. -tion, -ment, -ness, -ity, -ive, etc.)
- Grammatical error ratio
- Misspelled word ratio
- Word count
- MTLD (Measure of Textual Lexical Diversity)

**Parameters**
- 55 parameters
-    50 weights
-    5 biases

**Libraries used:**

- NumPy
- Pandas
- NLTK (for text parsing)
- Matplotlib
- os
- jdk
- spellchecker (misspelling checker)
- language_tool_python (grammar checker)
- multiprocessing (parallelism)

**Predict Your Own Level in predictor.ipynb**
When ran, you will be given a random prompt to answer.
After answering, it will output your predicted CEFR level.