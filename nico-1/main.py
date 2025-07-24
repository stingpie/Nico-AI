import torch as T
import torch.nn.functional as F
import os
from sdapm import SDAPM
from languagemodel import LanguageModel

if not os.path.exists("gutenberg_conv.json"):
  from datasets import load_dataset
  ds = load_dataset("willwade/Gutenberg-dialog-en")

  ds.set_format("torch")

  import json
  print(ds['train']['text'][:10])
  data = ("`GO`"+("`STOP`\n`GO`".join(ds['train']['text']))+"`STOP`").split("\n`GO``STOP`\n")
  with open("gutenberg_conv.json","w") as f:
    json.dump(data, f)
  print(data[0])
else:
  import json
  with open("gutenberg_conv.json",'r') as f:
    data  = json.load(f)
  print(data[0])

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer


vocab_size = 6400

if not os.path.exists("unigram-"+str(vocab_size)+".json"):


  import tokenizers.normalizers as nz

  tokenizer = Tokenizer(Unigram())
  tokenizer.normalizer=nz.Lowercase()

  import tokenizers.pre_tokenizers as pretok
  tokenizer.pre_tokenizer = pretok.Sequence([pretok.Digits(),pretok.Punctuation(), pretok.Split(" ", 'isolated')])

  trainer = UnigramTrainer(vocab_size=vocab_size, unk_token="`UNK`", special_tokens=["`UNK`","`GO`", "`PAD`", "`STOP`"])
  print("training... (tokenizer)")
  import random
  tokenizer.train_from_iterator(random.sample(data,10_000), trainer)

  tokenizer.save("unigram-"+str(vocab_size)+".json")
else:
  tokenizer = Tokenizer.from_file("unigram-"+str(vocab_size)+".json")


if not os.path.exists("tokenized_set.pkl"):
  print("tokenizing...")
  import random
  dataset = tokenizer.encode_batch(random.sample(data, len(data)//2))
  with open("tokenized_set.pkl", 'wb') as f:
    import pickle
    pickle.dump(dataset, f)
else:
  import pickle
  dataset = pickle.load(open("tokenized_set.pkl",'rb') )
print(dataset[0].tokens)



import torch_optimizer as optim

sdapm=None
vae=None
loss=None
vae_opt=None
sdapm_opt=None
import gc
gc.collect()
T.cuda.empty_cache()
gc.collect()


batch_size=3
emb_size = 512
sdapm_emb_size = 56
epoch=0
iter=0
count=0
device='cuda'

checkpoints = list(filter(lambda x: x.startswith('TVAT'), os.listdir()))
if checkpoints==None or len(checkpoints)==0:
    vae = LanguageModel(emb_size, sdapm_emb_size, vocab_size, temporal_division=64, device=device)
    vae.tvat.encoder.expansion_offset=0
else:
    print("loading", sorted(checkpoints)[-1])
    vae = T.load(sorted(checkpoints)[-1], weights_only=False)


checkpoints = list(filter(lambda x: x.startswith('sdapm'), os.listdir()))
sdapm = SDAPM(sdapm_emb_size, device=device)
if not ( checkpoints==None or len(checkpoints)==0):
    print("loading", sorted(checkpoints)[-1])
    sdapm.load_state_dict(T.load(sorted(checkpoints)[-1]) )


vae_opt = optim.DiffGrad(vae.parameters())
sdapm_opt = optim.DiffGrad(sdapm.parameters())

max_grad_norm=100
def grad_norm(grad):
    global max_grad_norm
    if(T.any(grad.abs()>max_grad_norm)):
        return F.normalize(grad, dim=-1)*max_grad_norm
    return grad




for p in sdapm.parameters():
    p.register_hook(grad_norm)
for p in vae.parameters():
    p.register_hook(grad_norm)

import time


while True:

  count = count+batch_size % (len(dataset)-batch_size)
  data_batch = T.nn.utils.rnn.pad_sequence([T.tensor(data.ids) for data in dataset[count:count+batch_size]], batch_first=True, padding_value=tokenizer.token_to_id("`PAD`"), padding_side='right').int().to(device)


  if(iter<300 or iter%30<15):

    reconstruction, len_loss = vae(data_batch)
    loss = F.cross_entropy(reconstruction.transpose(-1,-2), data_batch.long()) + vae.kl/emb_size/batch_size + T.mean(len_loss)
    loss.backward()
    vae_opt.step()
    if(iter%100==0):
      print("---===+++ EXAMPLE RECONSTRUCTION +++===---\n"+tokenizer.decode(T.argmax(reconstruction[0], dim=-1).tolist())+"\n---===+++#############+++===---")
  elif(iter%30>=29):

    embedded, len_loss, sections = vae.embed(data_batch[:,:-1])
    #print(T.histc(T.floor(sections)[:,-1])
    z = sdapm(embedded)
    prediction= vae.exbed(z, vae.embedding(data_batch[:,:-1]))
    loss = F.cross_entropy(prediction.transpose(-1,-2), data_batch[:,1:].long())
    loss.backward()
    sdapm_opt.step()
    vae_opt.step()
    print("---===+++ EXAMPLE RESULT +++===---\n"+tokenizer.decode(T.argmax(prediction[0], dim=-1).tolist())+"\n---===+++#############+++===---")
  else:

    enc_data = vae.embed(data_batch)[0].detach()
    prediction, r_loss, w_loss = sdapm(enc_data[:,:-1], enc_data[:,1:])
    loss = F.mse_loss(prediction, enc_data[:,1:]) + r_loss + w_loss
    loss.backward()
    sdapm_opt.step()
    #print(tokenizer.decode(T.argmax(prediction[0], dim=-1).tolist()))

  iter+=1
  vae_opt.zero_grad(set_to_none=True)
  sdapm_opt.zero_grad(set_to_none=True)
  gc.collect()
  print("EPOCH "+str(math.floor(iter/len(dataset))) +"\tLOSS: "+ str(loss.item()))
  if(iter%500==0):
    current_time = time.time()
    T.save(sdapm.state_dict(), "sdapm_"+str(int(current_time))+".pt")
    T.save(vae, "TVAT_"+str(int(current_time))+".pt")



