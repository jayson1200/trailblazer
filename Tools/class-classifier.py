import sys
import copy

sys.path.insert(1, f'/home/meribejayson/Desktop/Projects/trailblazer/llama')
from llama import Llama, Dialog

from typing import List

prompt = """Say there are the following categories of classes:
Art
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
Not Specific

I am going to give you a class name at Stanford University. Using the class name, output a list of categories, from the list above, tht describe that class at Stanford University.

For example:

CS 229:
Artificial Intelligence

CS 29N
Artificial Intelligence, Theory

CS 40:
Systems, Software Engineering
"""

"""
Command to run:

torchrun --nproc_per_node 1 class-classifier.py
"""

base_dialog = [
    {
        "role": "system",
        "content": prompt,
    },
    {
        "role": "user",
        "content": "CS 229: Machine Learning (STATS 229)",
    },
    {"role": "assistant", "content": "Artificial Intelligence"},
    {
        "role": "user",
        "content": "CS 29N: Computational Decision Making",
    },
    {"role": "assistant", "content": "Artificial Intelligence, Theory"},
    {
        "role": "user",
        "content": "CS 40: Cloud Infrastructure and Scalable Application Deployment ",
    },
    {"role": "assistant", "content": "Systems, Software Engineering"},
]

classes = ["CS 108: Object-Oriented Systems Design", 
           "CS 193P: iOS Application Development", 
           "CS 24: Minds and Machines (LINGUIST 35, PHIL 99, PSYCH 35, SYMSYS 1, SYMSYS 200)"
           "CS 83N: Playback Theater",
           "CS 103: Mathematical Foundations of Computing",
           "CS 106B: Programming Abstractions"]


ckpt_dir = f"/home/meribejayson/Desktop/Projects/trailblazer/llama/llama-2-7b-chat"

tokenizer_path = f"/home/meribejayson/Desktop/Projects/trailblazer/llama/tokenizer.model"
temperature: float = 0
top_p: float = 0
max_seq_len: int = 4096
max_gen_len: int = 512
max_batch_size: int = 4

def create_dialog(class_name: str) -> Dialog:
    new_dialog = copy.deepcopy(base_dialog)

    new_dialog.append({
        "role": "user",
        "content": class_name,
    })

    return new_dialog

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


dialogs: List[Dialog] = [create_dialog(c) for c in classes]

results = generator.chat_completion(
    dialogs,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
)


for dialog, result in zip(dialogs, results):
    print(result['generation']['content'])