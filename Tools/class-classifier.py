import sys

sys.path.insert(1, f'/home/meribejayson/Desktop/Projects/trailblazer/llama')
from llama import Llama

from typing import List

prompt = """
Say there are the following categories of classes:

Computer Engineering
Theory
Artificial Intelligence
Algorithms
Computational Biology
Graphics
Human Computer Interaction
Information
Systems
Software Engineering
Robotics
Ethics
Not specific

I am going to give you a class description. Using the class description, output a list of categories that fit that description.

For example:
(Only one of 173A or 273A counts toward any CS degree program.) A coder's primer to Computational Biology through the most amazing "source code" known: your genome. Examine the major forces of genome "code development" - positive, negative and neutral selection. Learn about genome sequencing (discovering your source code from fragments); genome content: variables (genes), control-flow (gene regulation), run-time stacks (epigenomics) and memory leaks (repeats); personalized genomics and genetic disease (code bugs); genome editing (code injection); ultra conservation (unsolved mysteries) and code modifications behind amazing animal adaptations. Course includes primers on molecular biology and text processing. Prerequisites: comfortable coding in Python from the command line.

Output:
Computational Biology

This course is designed to help students understand the unique challenges of solving security problems at scale, and is taught by senior technology leaders from companies tackling hardware and software security for hundreds of millions of people. The course is split into six parts covering major themes: Basics, Confidential Computing, Privacy, Trust, Safety and Real World. The format of the class will include guest lectures from experts in each theme, covering a blend of both theory and real world scenarios. Prerequisite: CS110/ CS111. Recommended but not required: CS155.

Output:
"""

"""
Command to run:

torchrun --nproc_per_node 1 class-classifier.py
"""


ckpt_dir = f"/home/meribejayson/Desktop/Projects/trailblazer/llama/llama-2-7b"

tokenizer_path = f"/home/meribejayson/Desktop/Projects/trailblazer/llama/tokenizer.model"
temperature: float = 0
top_p: float = 0
max_seq_len: int = 4096
max_gen_len: int = 15
max_batch_size: int = 4



generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

prompts: List[str] = [
    prompt
]

results = generator.text_completion(
    prompts,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
)

for prompt, result in zip(prompts, results):
    print(f"> {result['generation']}")