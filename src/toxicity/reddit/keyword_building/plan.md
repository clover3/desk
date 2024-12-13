
Goal: Make a classifier based on NLU features.

NLU features 
* example:
  * Is this text about policits?
  * Does this text contain urls?
  * Item

# Pipeline 1

* Generate keywords from each rules
* Check if each keyword applies* to the text using GPT
  * Paraphrase to "This text contains/is/is considered {}"
  * Get "statements"
  * Ask GPT if each comment in training data match the statement
  * (TODO: Measure true rate.)
  * Ideally, we want to distil this function into smaller models.
* For each keyword(statement), compute if this is good feature for the mod classification.
  * Collect ones that are useful, form feature vector. 
  * Train shallow model. (Logistic Regression, Decision Tree, etc.)

# Limitations of Pipeline 1

* Features are limited to the keywords that are in the rules
  * Rules mostly contain high-level features
  * 