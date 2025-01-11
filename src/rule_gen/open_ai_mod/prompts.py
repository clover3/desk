
orig = """
You are a user on Reddit browsing comments
on {}. Below is a comment from a Reddit
conversation in r/{} that received a high
number of upvotes.
Answer in a brief comma-separated list with
either one, two, or three keywords that summarizes
certain traits or qualities within the text
that might have led to the comment being highly
upvoted. If you think only one or two keywords
are necessary, then only include that many in
your response.
If for some reason you think this comment should’ve
instead been downvoted, answer with ‘N/A’.
Here’s the comment: {}
Your list:
"""

why_unsafe = """
Below is a text that is categorized unsafe as
 inputs or outputs of LLMs.
Answer in a brief comma-separated list with
either one, two, or three keywords that summarizes
certain traits or qualities within the text
that might have led to consider the text unsafe.
If you think only one or two keywords
are necessary, then only include that many in
your response.
If for some reason you think this text should 
not have been classified unsafe.
Answer with ‘N/A’.
Here’s the text: {}
Your list:
"""


why_unsafe_key = """
Below is a text that is categorized unsafe as
 inputs or outputs of LLMs.
Guess why the text is consider the text unsafe.
Answer in a short phrase.
If for some reason you think this text should 
not have been classified unsafe.
Answer with ‘N/A’.
Here’s the text: {}
Your list:
"""