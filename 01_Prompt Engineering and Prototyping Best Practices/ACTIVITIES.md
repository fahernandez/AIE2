##### 🏗️ Activity #1:

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner. 
    - Aspect Tested: R/ The application can follow instructions-The question was answered correctly, but there was a lot of Markdown formatting that was not respected. A possible improvement is to improve the rendering of answers.
2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested: R/ The application provides a decent UX experience-When providing the paragraph, the application didn't support ctrl+enter to separate the instructions from the content. A possible improvement is to give support to ctrl+enter.
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested: R/ The application followed more specific instructions-The model followed instructions and the final story has a length in the range of 100-150 words (148).
4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested: R/ The application applies reasoning steps-The application answered the question and provided reasoning steps.
5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested: R/ The application can follow user instructions-The task was executed successfully, and the answer was of similar size to the provided paragraph. 

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

##### 🧑‍🤝‍🧑❓ Discussion Question #1:

What are some limitations of vibe checking as an evaluation tool?

I see two main disadvantages:
- Reproducibility: As the answers are not 100% deterministic, reproducing results might be challenging. Without reproducibility, it is difficult to be explicit about expected behaviors, and that makes communication difficult between the tester and developer.
- Ambiguity: Vibe check questions can be interpreted differently between different evaluators. Hence, arriving at consensus about the "right" behavior might be challenging.


##### 🚧 Advanced Build
I improved the markdown support - now when the API returns Markdown, the Markdown is rendered better. Proof of the improvement was submitted in the form as supporting images.