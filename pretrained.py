import transformers
from transformers import AutoModel, BertTokenizer, RobertaTokenizer
from transformers import BertConfig, BertModel
# 1.使用 PreTrainedModel.from_pretrained() 提前下载文件：
mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
mol_encoder = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False) #do_lower_case = true 忽略大小写 
prot_encoder = AutoModel.from_pretrained("Rostlab/prot_bert")

# 2.使用 PreTrainedModel.save_pretrained()，将文件保存到指定目录：

mol_tokenizer.save_pretrained("./pretrained/ChemBERTa-tokenizer")
mol_encoder.save_pretrained("./pretrained/ChemBERTa-encoder")

prot_tokenizer.save_pretrained("./pretrained/prot_tokenizer")
prot_encoder.save_pretrained("./pretrained/prot_encoder")