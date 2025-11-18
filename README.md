# Projects

Public final projects for *Intermediate Data Programming* (CSE 163). Additional projects can be submitted via pull request.

Predict your CEFR level by answering given questions

**Model trained on 9 features:**

- Number of sentences
- Average sentence length
- Standard deviation of the sentence lengths
- Average character count per word
- Content word proportion (e.g. nouns, verbs, adjectives, adverbs)
- Count of derivational morphology (e.g. -tion, -ment, -ness, -ity, -ive, etc.)
- Grammatical error ratio
- Misspelled words ratio
- Word count
- MTLD (Measure of Textual Lexical Diversity)
- 
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