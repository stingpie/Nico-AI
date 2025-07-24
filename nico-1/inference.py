import torch as T
import torch.nn.functional as F
import os
from sdapm import SDAPM
from languagemodel import LanguageModel
from tvat import TVAT
from tvat import BernoulliMix
from tvat import TVET
from tvat import TVDT
from tvat import PositionalEncoding

from tokenizers import Tokenizer
from tokenizers.models import Unigram


T.set_default_device('cuda')

vocab_size = 6400
sdapm_emb_size=64
if not os.path.exists("unigram-"+str(vocab_size)+".json"):
    raise OSError("Vocab file not found!")
else:
  tokenizer = Tokenizer.from_file("unigram-"+str(vocab_size)+".json")



device='cuda'

checkpoints = list(filter(lambda x: x.startswith('TVAT'), os.listdir()))
if checkpoints==None or len(checkpoints)==0:
    raise OSError("TVAT file not found!")
else:
    print("loading", sorted(checkpoints)[-1])
    vae = T.load(sorted(checkpoints)[-1], weights_only=False)


checkpoints = list(filter(lambda x: x.startswith('sdapm'), os.listdir()))
sdapm = SDAPM(sdapm_emb_size, mem_len=128, param_mem_mats=True, autoscale=True,  device=device)
if not ( checkpoints==None or len(checkpoints)==0):
    print("loading", sorted(checkpoints)[-1])
    sdapm.load_state_dict(T.load(sorted(checkpoints)[-1]) )
else:
    raise OSError()



for module in vae.modules():
    module.to(device)
    module.eval()
    module.training=False

for module in sdapm.modules():
    module.to(device)
    module.eval()
    module.training=False

vae.tvat.decoder.b_mix.set_probs(1)
vae.tvat.encoder.expansion_offset=0

penalty=0

z=None

with T.autograd.grad_mode.inference_mode():
    input_text=""
    while True:
        in_text = input("Type in text:")
        input_text += "`GO`"+in_text+"`STOP`\n"
        generated_text="`GO`"
        data = T.tensor(tokenizer.encode(input_text + generated_text).ids, device=device).unsqueeze(0)
        _, _, sections =vae.embed(data)
        section_nums=T.floor(T.max(sections))
        for i in range(100):
            data = T.tensor(tokenizer.encode(input_text + generated_text).ids, device=device).unsqueeze(0)

            if(i>0): ## only feed in the next section, as defined by the encoder.
                _, _, sections =vae.embed(data)
                if(T.floor(T.max(sections))>section_nums):
                    #print(data.size(), sections.size())
                    #print(section_nums, sections)
                    #print( (T.floor(sections)==section_nums).nonzero()+1)
                    data = data[:,: (T.floor(sections)==section_nums+1).nonzero()[0][1]]

            embedded, _, sections =vae.embed(data)

            #section_nums=T.floor(T.max(sections))
            z = sdapm(embedded)
            #z=embedded
            #z = F.normalize(z-T.mean(z,dim=-1, keepdim=True), dim=-1)

            print(z)
            prediction = vae.exbed(z, data.size(-1)+100, vae.embedding(T.cat((data,T.full((1,100), tokenizer.token_to_id("`PAD`"), device=device)), dim=-1).int())[:,1:])
           
            print(prediction[0,0], T.argmax(prediction,dim=-1))
            frequency_penalty = -T.log((T.sum(F.one_hot(data.long(), num_classes=vocab_size), dim=-2)+2.71828))*penalty

            probabilities = F.softmax(prediction[0]+frequency_penalty, dim=-1)
            print(probabilities[0],T.argmax(probabilities, dim=-1))
            #probabilities[:,2]=0

            #frequency_penalty = 1/(T.sum(F.one_hot(data.long(), num_classes=vocab_size), dim=-2)+1)
            #probabilities*=frequency_penalty

            #absolute_penalty = T.where(T.sum(F.one_hot(data.long(), num_classes=vocab_size), dim=-2)>20, 0, 1)
            #probabilities*=absolute_penalty

            
            generated_text += str(tokenizer.decode(T.argmax(probabilities[data.size(1):], dim=-1).tolist())).replace("   ","`~`").replace(" ","").replace("`~`"," ")
            total_text = str(tokenizer.decode(T.argmax(probabilities, dim=-1).tolist())).replace("   ","`~`").replace(" ","").replace("`~`"," ")
            print("---------================-----------")
            print(total_text)# str(tokenizer.decode(T.argmax(probabilities[data.size(1):], dim=-1).tolist(), skip_special_tokens=False)).replace("   ","`~`").replace(" ","").replace("`~`"," "))
            break
            if("`STOP`" in generated_text):# or "\n" in generated_text):
                input_text +=generated_text[:generated_text.index("`STOP`")+6]+"\n"
                break
#            if("\n" in generated_text):# or "\n" in generated_text):
 #               input_text +=generated_text[:generated_text.index("\n")+1]
  #              break
        print(input_text)
        break
        


