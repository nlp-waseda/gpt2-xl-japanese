from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=50000,
    min_frequency=1,
    special_tokens=["<unk>", "<s>", "</s>"],
    limit_alphabet=8000,
    )
files = ["/home/mdxuser/data/wiki_mrph.txt", "/home/mdxuser/data/cc100_mrph.txt"]
tokenizer.train(files, trainer)
tokenizer.save("juman-bpe-wiki-cc100-50000.json")
