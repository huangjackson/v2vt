import ctranslate2
import sentencepiece as spm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Temporarily hard-coded paths, only en-zh
translator = ctranslate2.Translator('models/en-zh', device=device)
sp = spm.SentencePieceProcessor('models/en-zh/source.spm')

# Temporary placeholder
input_text = 'I want to wish you all a very happy Thanksgiving!'
input_tokens = sp.encode(input_text, out_type=str)

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = sp.decode(output_tokens)

print(output_text)
