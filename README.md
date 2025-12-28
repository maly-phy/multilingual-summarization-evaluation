# Multilingual Summarization Evaluation
<<<<<<< HEAD



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.gwdg.de/mohamed.aly/multilingual-summarization-evaluation.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.gwdg.de/mohamed.aly/multilingual-summarization-evaluation/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
=======
This is the official repository of the multilingual summarization evaluation project for the practical seminar `selected topics in data science` at Giplab (University of Göttingen). In this project, we run multiple experiements to generate and assess the quality of meeting summaries from different perspectives. We use standard NLP metrics (count-based), semantic, and LLM-as-judge evaluation metrics. We apply a State-of-the-Art (SOTA) method to extract the atomic facts from the meeting transcripts, which are then used to review and correct the given reference summary, and generate a new summary based on the oroginal meeting transcript and the atomic facts. We find that the atomic facts increase the accuracy and correctness of the reference summary drastically and also summaries generated are more factually consistent than the ones generated by relying only on the meeting transcript.  
We measure the consistency of results between English and German for all evaluation metrics to see how the scores change between languages.  I 

## Installation
```bash
git clone https://gitlab.gwdg.de/mohamed.aly/multilingual-summarization-evaluation.git
cd multilingual-summarization-evaluation
pip install -r requirements.txt
```

## Data
We use the synthesis dataset from [You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations](https://aclanthology.org/2025.findings-acl.599/)
| Parameter Count | English | German|
|:---------------:|:-------:|:-----:|
| **Meetings** | 500 | 261 |
| **Meeting per Type** | stakeholder: 57 <br/> innovation: 77 <br/> brainstorming: 87 | stakeholder: 41 <br/> innovation: 33 <br/> brainstorming: 31 |
| **Avg participants** | 4.47 | 4.3 |
| **Avg turns per meeting** | 82 | 70 |
| **Avg words per meeting** | 2395 | 2013  |
| **Vocab size** | 18852 | 19188 |
| **Avg summary length (words)** | 171.5 | 149.3 |

We run our experiments on only 30 rows of each corpus due to the limitations of computation resources and the rate limit of the API key we use for the model. In this work, we are more interested in investigating different evaluation methods than waiting a long time to execute on the full corpus of each language. 

## Model
We use the `llama-3.1-8b-instant` model which we run it using the Groq API key. To call the model, you need to add your API key in the .env file:  
```bash
export GROQ_API_KEY="{your-api-key}"
```

## Methodology
### Initial Summary Generation & Evaluation
We start by prompting the model to generate an initial summary for each language, based on a given meeting transcript that represents a discussion betwwen a group of people in certain topics. The summary generated has a maximum length of 250 words. We then evaluate the summary using the following metrics: 
<details>  

**Rouge**: it is a count-based metric that compares word overlap between the predicted generated summary and the reference. We use 1-2 grams and the longest common subsequence, and compute the precision, recall and f1-score for each rouge type.  
**Bleu**: similar to rouge but focuses on precision (matched words from the candidate to the reference summary).  
**Meteor**: it is based on harmonic mean between precision and recall. We use the appropriate word tokenizer for each language from NLTK.
**CHRF**: it calculates the similarity using character n-grams instead of word n-grams which is suitable for high morphology language like German. It also accounts for the F-score which is more convenient.
**Perplexity**: it measures the probability the model predicts the next word, quantifying the model's surprise. It cannot be compared across datasets of different vocab size as in our case. The lower perplexity the better.  
**Sacrebleu**: it is extended implementation of the bleu metric that provides additional features. We use the `spBLEU-1K` tokenizer which is more robust for multilingual text.  
**Bleurt**: it measures how the candidate text is fluent and conveys the meaning of the reference. It is a trained metric based on BERT model, and can be finetuned to specific data. We implement it from its [official repository](https://github.com/google-research/bleurt). We use the pretrained checkpoints [BLEURT-20-D12](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip) which was tested on 13 languages including English and German.  
**BERT**: it measures the semantic similarity between the reference and candidate summary. For the English text, we use the defult `bert-base-uncased` model, and the specialized `bert-base-german-cased` for the German.
**BART**: it combines bidirectional encoder (like BERT) with auto-regressive decoder (like GPT) which can not only understand text as in BERT but also generate it. The BART score is negatve since it calculates the the average log probability (0-1) of the generated tokens which yields negative. We use `bart-large-cnn` checkpoints for English and `mbart-large-cc25` for German.  
**LAR**: it measures the ratio of the length reference to generated summary normalized by the reference.  
**questeval**: it use a QA model to generate questions from the source transcript and the model generates answers back from the generated summary. Then a score is calculated to measure how much information the generated summary contains relative to the source. We used questeval for both languages, however the pipeline is still not adjusted for multilinguality.  
**LongDocFact**: we measure the factuality of the summary following the approach from [Bishop, Xie, et al.](https://aclanthology.org/2024.lrec-main.941.pdf). We measure the cosine similarity between each sentence in the generated summary and the sentences from the source (meeting transcript) and select the most closest sentence. Then, we take a context window of 5 preceding and 5 subsequent sentences from the most relevant sentence in the source. In the paper they took a context window of 3, but we found that increasing the size to 11 (more surrounding context) gives better scores to have. Then we compute the BART score between the sentence from the generated summary and the ones from the source, and averge the results.  
**Blanc**: it measures how well the summary helps a language model fill the blanks in its tasks; like predicting masked words. The model first performs tasks on the source document and the accuracy is calculated. Then, the generated summary is added to help the model and the boost of the model's performance is the blanc score. We use the models specific to each language as above in BERT. We additionally measure other features in the summary which are coherence, soft, and alarms. Coherence evaluates the logical structure and flow, while soft measures the variability in the summary. The alarms is used to give penality when hallucinated content or specific type of error is found (the lower the better).
**Lens**: it measures the text simplicity by comparing to a more complex text and a reference. We use the pre-trained checkpoints [davidheineman/lens](https://huggingface.co/davidheineman/lens). The higher the score, the more simple the text is. In our case, we will prefer the more complex and variant summary, which then becomes the lower score.  
The script that has all metrics is `src/compute_nlp_metrics.py`.
</details>

### Evaluating the Meeting Transcript
We use LLM-as-judge to evaluate the meeting transcript based on the likert scale (1-5). The evaluation metrics are taken from [You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations](https://aclanthology.org/2025.findings-acl.599/).  
**Naturalness**: it measures the flow of converstaion and if it is at the level of native speakers.  
**Coherence**: How the logical flow and connection is maintained in the converstaion.  
**Interesting**: it measures the richness of the content and variations in the converstaion.  
**Consistency**: if the speakers have equal contributions in the discussion.  
We also add other metrics like speaker dynamics (turns), coreference (who or what a pronoun refers to), and others wich can be found in `src/meeting_challenges_evaluator.py`.

### Extraction of Atomic Facts
We apply the `llama-3.1-8b-instant` model to extract the atomic facts from the meeting transcript in both languages. We then use the atomic facts with the meeting source to regenerate a more factual and accurate summary. In this way we give the model a more strict and bound key points to focus on rather than giving all meeting transcript whch is quite lengthy and model might get confused and distracted between the transitions in the discussion.  
We also noticed that the reference summary given in the data has hallucinated content that is not mentioned in the source meeting. For that we use our ground truth atomic facts to review the summary, determine the hallucinated content and either correct them (if the information exists in the atomic facts) or remove them completely. We recalculate all summary metrics and compare with the baseline of the initial summary. Although the atomic facts boosts the accuracy and correctness of the summary drastically, we find that the facts extracted from the German corpus are much less the ones from English. This is indeed due to the model's understanding capability when it comes to other languages than English. The script of atomic facts is `src/fact_score.py`.  

### Criteria Evaluation of Summary
We assess the linguistic (grammar norms and structure) and naturalness aspects of the summary. We also evaluate the factuality of the summary based on the atomic facts. For the German summary we evaluate the factuality based on the source meeting solely, since the extracted facts are shorter compared to English. This avoid the german summary being penalized if the model doesn't find the information in the facts however they exist in the source meeting. The script used is `src/summary_criteria_scoring.py`.  

### Human Correlation
We estimate the correlation with the model scores of the initial summary (without atomic facts) for the naturalness criteria. A native German speaker assesses the German summary based on the Likert scale (1-5) and non-native English speaker assesses the English one. We find that the German correlation with the model is higher than its counterpart in English. We calculate the `spearman`, `pearson` and `kendall` correlation metrics.  

### Multilingual Consistence
We estimate the model's consistence scoring between languages, which is basically that if the model judgement is consistent across languages, then we expect it to give the same evaluation score to all multilingual summaries if they contain exactly the same content. The model's knowledge of English is excellent but then it degrades to other languages, especially the low source ones which we don't cover in this work.   
We calculate the SOTA `Language Consistence Index` (LCI) [Son, et al.](https://arxiv.org/pdf/2410.17578) for all evaluation metrics of the summary, given an accepted summary (regenerated with help of atomic facts) and a rejected one (initial without atomic facts). We take the difference between average scores of the rejected and accepted summaries for each language as follows:  
$$
\mathrm{LCI} = \frac{1}{N} \sum_{i=1}^{N} \frac{\Delta S_i}{\Delta S_{\text{norm}}}
$$
$$
\Delta S_{\text{norm}} = \max_i \Delta S_i
$$
where $N$ is the total number of languages. If the model is fairly consistent, then $LCI$ would be one. The script of this section is `src/evaluate_consistency.py`.

## Results
### Summary Evaluation (NLP Metrics)
Here we present the evaluation results of the initial and regenerated summaries in both languages for the NLP metrics. The numbers represent the mean value.  
<details>

| Metric | Initial (ENG) | Regenerated (ENG) | Initial (GER) | Regenerated (GER) |
| :-----------------: | :-----------: | :---------------: | :------------:| :----------------:|
| **Rouge1 F1** | 0.373 $\pm$ 0.05 | **0.54** $\pm$ 0.09 |0.28 $\pm$ 0.04| **0.57** $\pm$ 0.12| 
| **Rouge2 F1** | 0.084 $\pm$ 0.08 | **0.28** $\pm$ 0.11 |0.05 $\pm$ 0.02| **0.38** $\pm$ 0.16|
| **RougeL F1** | 0.22 $\pm$ 0.03 | **0.44** $\pm$ 0.11 |0.17 $\pm$ 0.03| **0.48** $\pm$ 0.14|
| **Bert precision** | 0.612 $\pm$ 0.03 | **0.70** $\pm$ 0.05 |0.64 $\pm$ 0.03| **0.77** $\pm$ 0.05|
| **Bert recall** | 0.60 $\pm$ 0.03 | **0.69** $\pm$ 0.05 |0.62 $\pm$ 0.03| **0.77** $\pm$ 0.07|
| **Bert F1** | 0.60 $\pm$ 0.03 | **0.69** $\pm$ 0.05 |0.63 $\pm$ 0.03| **0.77** $\pm$ 0.06|
| **LAR** | 0.81 $\pm$ 0.15 | **0.85** $\pm$ 0.17 |0.87 $\pm$ 0.11| 0.87 $\pm$ 0.12|
| **Blanc help** | 0.13 $\pm$ 0.02 | **0.31** $\pm$ 0.07 |0.39 $\pm$ 0.03| **0.71** $\pm$ 0.09|
| **Blanc tune** | 0.08 $\pm$ 0.02 | **0.23** $\pm$ 0.07 |0.05 $\pm$ 0.02| **0.29** $\pm$ 0.09|
| **Bleurt** | 0.41 $\pm$ 0.05 | **0.42** $\pm$ 0.06 |0.33 $\pm$ 0.06| **0.42** $\pm$ 0.1|
| **Meteor** | 0.27 $\pm$ 0.05 | **0.41** $\pm$ 0.08 |0.19 $\pm$ 0.02| **0.46** $\pm$ 0.12|
| **Bleu** | 0.04 $\pm$ 0.03 | **0.23** $\pm$ 0.11 |0.01 $\pm$ 0.02| **0.34** $\pm$ 0.16|
| **Chrf** | 44.6 $\pm$ 4.03 | **57** $\pm$ 7.1 |39 $\pm$ 2.8| **63** $\pm$ 9.6|
| **Perplexity** | 21.3 $\pm$ 4.1 | **18** $\pm$ 5.9 |**29** $\pm$ 6.0| 30 $\pm$ 6.9|
| **Bart** | -3.17 $\pm$ 0.3 | **-2.6** $\pm$ 0.43 |-5.3 $\pm$ 0.5| **-3.7** $\pm$ 0.82|
| **LongDocFact** | -2.45 $\pm$ 0.22 | **-1.7** $\pm$ 0.40 |-5.1 $\pm$ 0.4| **-3.7** $\pm$ 0.49|
| **Lens** | 58.5 $\pm$ 6.06 | **58.2** $\pm$ 6.8 |52 $\pm$ 6.8| **46** $\pm$ 6.1|
| **Questeval** | 0.41 $\pm$ 0.05 | **0.52** $\pm$ 0.08 |0.31 $\pm$ 0.04| **0.41** $\pm$ 0.07|
| **Sacrebleu** | 8.44 $\pm$ 3.8 | **27.5** $\pm$ 10.8 |7.3 $\pm$ 2.6| **40** $\pm$ 16|
| **Estime alarms** | 87.06 $\pm$ 15.1 | **50.3** $\pm$ 18.9 |128 $\pm$ 17| **79** $\pm$ 29|
| **Estime soft** | 0.64 $\pm$ 0.05 | **0.81** $\pm$ 0.08 |0.25 $\pm$ 0.06| **0.6** $\pm$ 0.17|
| **Estime coherence** | 0.23 $\pm$ 0.12 | **0.36** $\pm$ 0.22 |0.1 $\pm$ 0.07| **0.38** $\pm$ 0.17|
</details>

### Evaluation of the Meeting Transcript (Criteria – LLM as Judge)
We presnt the results of evaluating the meeting transcript against its source article of each topic, based on the criteria mentioned above. We calculate the models' score and confidence for each language. The numbers represent the mean value.
<details>  

| Metric | Score (ENG) | Conf (ENG) | Score (GER) | Conf (GER) |
| :----: | :---------: | :--------: | :---------: | :--------: |
|**Naturalness**| **4.0** $\pm$ 0.0| 87 $\pm$ 5.3| 3.86 $\pm$ 0.4| 86 $\pm$ 4.5| 
| **Coherence**| **3.9** $\pm$ 0.23| 90 $\pm$ 5.1| 3.6 $\pm$ 0.56| 83 $\pm$ 4.7|
|**Interesting**| 4.0 $\pm$ 0.0| 93 $\pm$ 3| 4.0 $\pm$ 0.0| 90 $\pm$ 1.8| 
|**Consistency**| 3.9 $\pm$ 0.31| 92 $\pm$ 4.8| 3.9 $\pm$ 0.3| 82 $\pm$ 3.9|
|**Spoken language**| **3.3** $\pm$ 0.66| 80 $\pm$ 0.0| 2.6 $\pm$ 1.3| 80 $\pm$ 0.0|
|**Speaker dynamics**| 2.2 $\pm$ 1.0| 79 $\pm$ 4.5| **4.0** $\pm$ 0.0| 80 $\pm$ 0.0|
|**Coreference**| **3.2** $\pm$ 0.64| 80 $\pm$ 0.0| 2.9 $\pm$ 1.0| 80 $\pm$ 0.0|
|**Discourse structure**| 3.6 $\pm$ 0.67| 80 $\pm$ 0.0| **4.0** $\pm$ 0.0| 80 $\pm$ 0.0|
|**Contexual turn taking**| 3.8 $\pm$ 0.36| 80 $\pm$ 0.0| **4.0** $\pm$ 0.0| 80 $\pm$ 0.0|
|**Implicit context**| 3.8 $\pm$ 0.36| 80 $\pm$ 0.0| **4.0** $\pm$ 0.0 | 80 $\pm$ 0.0|
|**Low information density**| **3.4** $\pm$ 0.68| 80 $\pm$ 0.0| 3.3 $\pm$ 0.85| 80 $\pm$ 0.0|
</details>

### Summary Evaluation (Basic Criteria – LLM as Judge)
We evaluate the initial and regenrated summaries across languages, based on the basic criteria (naturalness, linguistic and factualness). We present the average score values of the model for each criteria.  
| Metric | Initial (ENG) | Regenerated (ENG) | Initial (GER) | Regenerated (GER) |
| :-----------------: | :-----------: | :---------------: | :------------:| :----------------:|
|**Linguistic**| 4.3 $\pm$ 0.13| **4.5** $\pm$ 0.17| **4.3** $\pm$ 0.2| 4.1 $\pm$ 0.27|  
|**Naturalness**| **4.0** $\pm$ 0.49| 3.9 $\pm$ 0.48| **3.9** $\pm$ 0.33| 3.6 $\pm$ 0.64|
|**Factuality**| 3.2 $\pm$ 1.3| **4.0** $\pm$ 0.59| 3.5 $\pm$ 0.59| **3.9** $\pm$ 0.46|

### Human Correlation
We present the human correlation with the models' scores for initial summaries for the naturalness criterion.
| Metric | Coeff (ENG) | P-value (ENG) | Coeff (GER) | P-Value (GER) |
| :----: | :---------: | :-----------: | :---------: | :-----------: |
|**Spearman**| -0.15 | 0.435| **0.22** | **0.26**|
|**Pearson**| -0.20 | 0.30 | **0.24**| **0.21**|
|**Kendall**| -0.13 | 0.44| **0.21**| **0.26**|

### Multilingual Consistency
We calculate the LCI for all metrics using the rejected and accepted summaries to see how well the model is consistent across languages.
| Metric | LCI | Metric | LCI |
| :----: | :-: | :----: | :-: |
|**Bart**|0.68| **Bleurt**| 0.57|
|**LongDocFact**| 0.76| **Meteor**| 0.75|
|**Questeval**| **0.91**| **Bleu**| 0.79|
|**Bert precision**| 0.82| **Chrf**| 0.76|
|**Bert recall**| 0.80| **Perplexity**| 0.69|
|**Bert F1**| 0.81| **Sacrebleu**| 0.79|
|**Blanc help**| 0.78| **LAR**| 0.50|
|**Blanc tune**| 0.81| **Rouge1 F1**| 0.79|
|**Estime alarms**| 0.87| **Rouge2 F1**| 0.80|
|**Estime soft**| 0.73| **RougeL F1**| 0.86|
|**Estime coherence**| 0.71| **Lens**| 0.52|
|**Linguistic**| 0.74| **Naturalness**| 0.57|
|**Factuality**| 0.71|                |     |


>>>>>>> c59f003 (adding a readme file for the project)
