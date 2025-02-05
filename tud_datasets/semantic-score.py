import torch
import math
import re
import textstat
from datasets import load_dataset
from enchant.checker import SpellChecker
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM


data_set = load_dataset("nbroad/basic_text_dataset", split="train")
spam_data_set = load_dataset("scholl99/spam_email_v0", split="train")

checker = SpellChecker("en_US")

deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
gpt_model_name = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

gpt_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(device)
gpt_model.eval()  


ds_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name, trust_remote_code=True)
ds_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_name,
    torch_dtype=torch.float16,  
    trust_remote_code=True
).to(device)



ds_model.eval()

text_dict = {
    "very_bad_quality": [
        "i want to say somethin about this topic but its hard to say the words. like you know when things is happening and u dont know? thats kinda like this. anyway i think its importan but who cares lol",
        "food is good. people eat it every day. i like pizza and burgers. sometimes food is bad tho. like when its cold and not tasty. but people need food to not be hungry. so food is important.",
        "Fast food is food that is fast. Many people eat it every day because it is easy. There are burgers, fries, and pizza. It tastes good but is not very healthy. Some people like it a lot and eat it all the time.",
        "Social media is everywere. Ppl use it al day to talk, post pic, and do stuff. It good for fun but also bad somtimes. Peple get mad easy and argue too much. Many ads also anoying. But stil, evryone use it so its inportant.",
        "Viddeo games are fun. Peple play them all the time. Some game are hard, some are easy. You can play alone or with frends online. Graffics are getting better every year. Some peple say games are bad for kids, but others say its ok. Games are just games.",
        "Pets are cute. Dog and cats are most poplar pets. They live in peples homes and make them happy. Some peple like brids or rabits too. You need to feed them and take care of them. Pets are nice but sumtimes messy."
    ],
    "bad_quality": [
        "The topic of history is something that people talk about a lot. There were a lot of wars and events in history that happened, and they changed things. Some people think history is interesting, but some do not. There are books about it, and people study it in school.",
        "The universe is big. There are many stars and planets. Some people think there are aliens, but nobody knows. Astronauts go to space, and they see Earth from far away. The moon is close, and people went there a long time ago. Space is interesting because it is huge.",
        "Eating heathy is good becuse it helps ur body. Vegtables and fruits are beter then junk food. Some peple eat too much shuger and that makes them sick. Protien is also inportant, like meat and eggs. Driking water is also good for helth. If you eat good, u feel good.",
        "Traveling is fun becuse you see new places. Peple go to diferent countrys for vaction or work. You can travel by plane, car or train. Some places have very good food and beutiful views. But traveling cost money and can be tired. Some peple like travling alone, some go with frends or family.",
        "Sport is good for health and make you strong. Peple play soccer, basketball, tennis and other sports. It helps you be fit and have fun. Some sports need alot of running, others not so much. Some peple play just for fun, others play to win. Sports also bring peple together in teams and events."
    ],
    "average_quality": [
        "Technology has changed the way people live their lives in many ways. With the rise of smartphones and the internet, communication has become faster and more convenient. However, some argue that people have become too dependent on technology, leading to decreased social interaction in person. While technology brings many benefits, it is important to find a balance between its use and real-life interactions.",
        "Climate change is a serious issue that affects the entire planet. Due to human activities such as burning fossil fuels, the Earth's temperature is rising. This has led to extreme weather patterns, melting ice caps, and rising sea levels. Scientists and environmentalists urge people to reduce their carbon footprint by using renewable energy sources, conserving water, and reducing waste. While some governments have taken action, more efforts are needed to slow down the effects of climate change.",
        "Getting enough sleep is essential for both physical and mental health. When people don’t sleep well, they can feel tired, irritated, and have trouble concentrating. Studies show that a lack of sleep can also lead to serious health problems, including heart disease and a weakened immune system. Experts recommend that adults get at least seven to eight hours of sleep each night. Developing a healthy sleep routine, such as avoiding screens before bedtime and keeping a consistent schedule, can help improve sleep quality.",
        "Reading books is a great way to expand knowledge, improve vocabulary, and enhance imagination. Books come in many genres, from fiction to non-fiction, allowing people to explore different perspectives and ideas. Studies suggest that reading regularly can improve focus, reduce stress, and even contribute to better sleep. While digital devices offer convenience, many readers still prefer physical books for the unique experience they provide. Developing a reading habit can be beneficial for both personal and professional growth.",
        "Mental health is just as important as physical health, yet it is often overlooked. Stress, anxiety, and depression can negatively impact a person’s daily life and overall well-being. Taking care of mental health includes activities like exercise, meditation, and seeking professional help when needed. Many people still face stigma when discussing their struggles, but awareness is growing, and more resources are available. Prioritizing mental health can lead to a happier and more productive life."
    ],
    "good quality": [
        "The advancements in artificial intelligence have revolutionized various industries, including healthcare, finance, and entertainment. Machine learning algorithms now assist in diagnosing diseases, optimizing stock market predictions, and even generating realistic visual content. However, ethical concerns arise regarding data privacy, job displacement, and the potential misuse of AI. Striking a balance between innovation and ethical considerations remains a critical challenge for society moving forward.",
        "The rise and fall of ancient civilizations offer valuable insights into human history. The Egyptians, Greek, and Mayans each developed advanced societies with unique contributions in architecture, writing, and governance. For example, the Egyptians built monumental pyramids as tombs for their pharaohs, while the Greek developed one of the earliest known writing systems, cuneiform. Despite their innovations, many ancient civilizations eventually declined due to environmental changes, internal conflicts, or invasions. Studying these societies helps historians understand patterns that continue to shape modern nations.",
        "Music has played a vital role in human culture for centuries, shaping traditions, emotions, and social connections. From classical symphonies to modern pop songs, music reflects the values and experiences of different societies. In many cultures, music is used in rituals, celebrations, and storytelling. Scientific research has also shown that music can influence mood and cognitive function, making it not just an art form but a powerful tool for communication and emotional expression. As technology advances, the ways people create and experience music continue to evolve, further enriching its impact on society.",
        "Architecture has evolved significantly throughout history, reflecting cultural, technological, and social changes. Ancient civilizations, such as the Greeks and Romans, built grand structures that still influence modern designs. During the Middle Ages, Gothic architecture introduced towering cathedrals with intricate details, while the Renaissance brought a revival of classical elements. In the modern era, architectural styles have become increasingly diverse, integrating sustainability and cutting-edge materials. As urbanization expands, architects continue to innovate, balancing functionality with aesthetic appeal.",
        "Social movements have historically played a crucial role in shaping societies and influencing policies. From the civil rights movement to modern environmental activism, these collective efforts aim to bring about social, political, or economic change. Movements often begin with grassroots organization and gain momentum through public demonstrations, media influence, and legal actions. While some movements achieve their goals, others face significant resistance. Regardless of the outcome, they highlight the power of collective action in driving societal progress."
    ],
    "excellent_quality": [
        "The rapid evolution of artificial intelligence has sparked both excitement and apprehension across multiple disciplines. From enhancing medical diagnoses to streamlining complex logistical operations, AI continues to push the boundaries of what was once thought possible. However, alongside these advancements come critical ethical concerns—ranging from algorithmic biases to the erosion of personal privacy. As industries rush to integrate AI into their workflows, policymakers and technologists alike must navigate the fine line between innovation and responsible implementation. The challenge lies not only in leveraging AI’s immense potential but also in ensuring its development aligns with human values and societal well-being.",
        "Time is one of the most fascinating and elusive concepts in philosophy. Throughout history, thinkers such as Aristotle, Augustine, and Einstein have debated whether time is an objective reality or merely a construct of human perception. In classical physics, time is treated as a linear and absolute entity, yet modern theories like relativity suggest it is fluid and influenced by gravity. Additionally, some philosophical perspectives argue that the past and future are illusions, and only the present moment truly exists. As scientific discoveries continue to reshape our understanding, the nature of time remains an open and profound question that challenges both physicists and philosophers alike.",
        "The rapid development of artificial intelligence has introduced complex ethical dilemmas that challenge both policymakers and society at large. While AI has revolutionized industries by automating tasks, optimizing efficiency, and enhancing decision-making, concerns surrounding privacy, bias, and accountability remain significant. For instance, biased algorithms can perpetuate societal inequalities, while the rise of AI-driven surveillance raises serious questions about individual freedoms. Moreover, as AI systems become more autonomous, determining responsibility in cases of errors or harm becomes increasingly difficult. Balancing innovation with ethical responsibility requires ongoing discussions, regulations, and interdisciplinary collaboration to ensure AI benefits humanity without compromising fundamental values.",
        "Human decision-making is a complex process influenced by cognitive biases, emotions, and external factors. Psychological research suggests that people often rely on heuristics—mental shortcuts that simplify choices but can lead to errors in judgment. For example, the confirmation bias causes individuals to favor information that supports their existing beliefs while ignoring contradictory evidence. Additionally, emotions play a critical role in decision-making, sometimes overriding rational analysis. Understanding these psychological mechanisms can help individuals and organizations make more informed and effective choices, reducing errors and improving outcomes in various aspects of life.",
        "Genetic engineering presents one of the most controversial ethical dilemmas of modern science. While advancements in gene editing have the potential to eradicate hereditary diseases and improve agricultural yields, they also raise concerns about unintended consequences and ethical boundaries. The debate extends beyond science, touching on issues of genetic modification in humans, designer babies, and potential inequalities in access to genetic enhancements. As technology continues to progress, societies must establish ethical frameworks that balance innovation with moral responsibility, ensuring that genetic advancements serve humanity without compromising fundamental ethical principles."
    ]
}

def is_url(word):
    # This regex will match strings like:
    # "hello.me", "http://example.com", "https://www.example.co.uk", etc.
    url_regex = re.compile(
        r'^(https?://)?'            # Optional protocol (http:// or https://)
        r'(www\.)?'                 # Optional "www."
        r'([a-zA-Z0-9-]+\.)+'       # One or more domain parts followed by a dot (e.g. hello.)
        r'[a-zA-Z]{2,}'             # Top-level domain (e.g. "me", "com", "org")
        r'(/[^\s]*)?$',             # Optional path
        re.IGNORECASE
    )
    return re.match(url_regex, word) is not None

import csv
from pathlib import Path

csv_filepath=Path('./abbreviations_eng.csv')

def load_abbreviations(csv_filepath=csv_filepath):
    abbreviations = set()
    with open(csv_filepath, newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            # Normalize the abbreviation (e.g., lower-case, strip whitespace)
            abbr = row['abbr'].strip().lower()
            abbreviations.add(abbr)
    return abbreviations


def calculate_perplexity(text, model, tokenizer):
    perplexity = 1000 # in case we fail, just give a very high perplexity score.
    try:
        # Encode the text into input IDs
        input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    
        # Get the loss; note that we pass the same input as labels to compute the loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss  # This is the average cross-entropy loss per token
    
        # Perplexity is defined as the exponential of the loss
        perplexity = math.exp(loss.item())
    except Exception as e:
        print(f"Error processing text: {text}")
        print(e)
    return perplexity


def lexical_diversity(text):
    return len(set(text)) / (0.5*len(text))

abbreviation_set = load_abbreviations()

def is_abbreviation(word, abbr_set=abbreviation_set):
    # Remove trailing punctuation (like a period) that might be attached to an abbreviation.
    normalized = word.strip('.,;:!?"\'').lower()
    return normalized in abbr_set

def count_spelling_errors(text, abbr_set=abbreviation_set):
    checker.set_text(text)
    num_err = 0
    for err in checker:
        word = err.word
        if not (word[0].isupper()) and not is_url(word):
            if is_abbreviation(word):
                num_err += 0.5
            else: 
                num_err += 1
    return num_err 




eps = 1e-11
def logistic_trans(raw, k, c):
    return 1 / (1 + math.exp(-k*(math.log(raw+eps)-c)))

def quality_score(spelling_errors, m1_pp, m2_pp, lex_divers, text_standard, verbose_level=0):
    alpha = 3 
    beta = 0.35
    gamma = 0.25

    if verbose_level > 1:
        print(f'spelling errors:{spelling_errors}, m1_pp:{m1_pp}, m2_pp:{m2_pp}, lex_divers:{lex_divers}, text_standard:{text_standard}')
    if  verbose_level > 2: 
        print(f'\tspelling errors:{spelling_errors}, log:{math.log(spelling_errors + eps)}, exp:{math.exp(spelling_errors)}, sqrt:{math.sqrt(spelling_errors)}, pwr:{spelling_errors**2}')
        print(f'\tm1 pp:{m1_pp}, log:{math.log(m1_pp)}, exp:{math.exp(m1_pp)}, sqrt:{math.sqrt(m1_pp)}, prw:{m1_pp**2}')
        print(f'\tm2 pp:{m2_pp}, log:{math.log(m2_pp)}, exp:{math.exp(m2_pp)}, sqrt:{math.sqrt(m2_pp)}, prw:{m2_pp**2}')
        print(f'\tlex divers:{lex_divers}, log:{math.log(lex_divers)}, exp:{math.exp(lex_divers)}, sqrt:{math.sqrt(lex_divers)}, prw:{lex_divers**2}')
        print(f'\ttext standard:{text_standard}, log:{math.log(text_standard)}, exp:{math.exp(text_standard)}, sqrt:{math.sqrt(text_standard)}, prw:{text_standard**2}')

    spelling_penalty = math.exp(beta * max(0, spelling_errors - 2))
    raw = ((text_standard**alpha) * lex_divers) / ((spelling_penalty) * gamma*math.sqrt(m1_pp*m2_pp))
    qual = logistic_trans(raw, k=0.45, c=0.5)
    if verbose_level:
        print(f'\traw: {raw}, score: {qual}')
    return qual 

    # return (math.log(text_standard) + math.log(lex_divers)) / (((spelling_errors + 1)**2) * ((m1_pp + m2_pp) / 2))

def filter_urls_from_text(text):
    # Split text into words/tokens
    tokens = text.split()
    # Filter out tokens that match our URL pattern
    filtered_tokens = [token for token in tokens if not is_url(token)]
    # Reconstruct the text from tokens
    return " ".join(filtered_tokens)


def calculate_score(text):
    m1_pp = calculate_perplexity(text, ds_model, ds_tokenizer)
    m2_pp = calculate_perplexity(text, gpt_model, gpt_tokenizer)
    lex_divers = lexical_diversity(text)
    text_stand = textstat.text_standard(text, float_output=True)
    text_no_urls = filter_urls_from_text(text)
    spelling_errors = count_spelling_errors(text_no_urls)
    return quality_score(spelling_errors, m1_pp, m2_pp, lex_divers, text_stand)


def run_example():
    for key in text_dict:
        print(key)
        for text in text_dict[key]:
            #print(text)
            print(f'\tscore: {calculate_score(text)}')

def construct_row(text):
    score = calculate_score(text)
    return {
        "text": text,
        "score": score
    }

def add_score(ds_item):
    score = calculate_score(ds_item['text'])
    return {'score': score}

def test_with_dataset():
    print(len(data_set))
    for i in range(50):
        text = data_set[i]['text']
        print(text)
        print(f'\tscore: {calculate_score(text)}')

def create_csv_from_dataset(data_set, filename):
    subset = data_set.select(range(25))
    subset_with_scores = subset.map(add_score)
    subset_with_scores.to_csv(f'{filename}.csv')


def construct_csv_from_dataset():
    test_datas = data_set[:25]

    for text in test_datas['text']:
        di = construct_row(text)
        print(di)


create_csv_from_dataset(spam_data_set, "spam")
