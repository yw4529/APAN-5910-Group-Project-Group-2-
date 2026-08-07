# Environment Setup Notes

If you're on an Intel Mac (x86_64), recent versions of PyTorch and Transformers
don't play well together out of the box. Here's what worked:

1. Pin Python to 3.11 (not 3.13):
   uv python pin 3.11
   uv venv --python 3.11

2. Pin torch to a version with Intel Mac wheels:
   uv add "torch==2.2.2"

3. Pin transformers to a version compatible with torch 2.2.2
   (transformers >=4.4 requires torch >=2.4, which has no Intel Mac wheels):
   uv add "transformers>=4.30,<5.0"

4. Pin numpy below 2.0 (torch 2.2.2 was built against NumPy 1.x):
   uv add "numpy<2"

5. Known bug: the saved tokenizer config specifies BertTokenizer, which
   generates a token_type_ids field that DistilBertForSequenceClassification
   does not accept. Fixed in inference.py by dropping this key before
   calling the model:
   enc.pop("token_type_ids", None)
