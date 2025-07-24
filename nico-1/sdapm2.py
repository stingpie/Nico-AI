


# automatic procedural memory
import torch as T
import torch.nn.functional as F
#import tensordict
import math, random

## altered from mitm4 to use a transformer as worth estimator.

@T.no_grad()
def pos_encoding(data_width, index):

    mult = 2**T.arange(data_width)
    sins = T.sign(T.sin(index* 2**T.arange(data_width//2)))
    coss = T.sign(T.cos(index * 2**T.arange(data_width//2)))

    return T.cat((sins,coss),dim=-1)



def softswish(x): ## an activation function similar to swish.
    b = 0.2974953
    return T.where(x>20,x-0.944031344, (1/b)* T.log(1+T.exp(b*x))*(x/(1+T.abs(x))))

def tapered_swish(x):
    softsign =(lambda x_: x_/(1+T.abs(x_)))
    return softsign(x)*(softsign(x+1)+1)



class SDAPM(T.nn.Module):

    def __init__(self, data_width, max_steps=10, mem_len=100, layer_gain=None, var_lr=False, param_mem_mats=False, autoscale=True, sparse_router='top 3', router_search = False, device = 'cpu'):

        super().__init__()

        self.device = device
        self.data_width = data_width
        self.max_steps = max_steps
        self.mem_len = mem_len

        if layer_gain==None:
            layer_gain=1/data_width
        self.layer_gain=layer_gain

        self.memories=None  ## I'm using both tensordict and nested tensors for this. Tensordict allows me to cleanly modify memories using a single index, and nested tensor allows for the addition of more memories as time goes on, without overwriting existing tensors.
        self.autoscale=autoscale
        atten_dat_size= data_width*9+1 + 2*int(math.ceil(math.sqrt(mem_len)))

        self.num_attn_heads=6
        self.attention_model = T.nn.TransformerEncoder(T.nn.TransformerEncoderLayer(math.ceil(atten_dat_size/self.num_attn_heads)*self.num_attn_heads, self.num_attn_heads, dim_feedforward=atten_dat_size, batch_first=True, device=self.device), 4)

        #self.worth_model = T.nn.LSTM(data_width*2, data_width*2*4, num_layers=3,  batch_first=True, device=self.device)
        self.worth_model = T.nn.LSTM(data_width*2, data_width*2*4, num_layers=3, batch_first=True, device=self.device)
        self.hx_cx = None

        self.initial_mem_mats = layer_gain*T.eye(self.data_width, device=self.device).expand(mem_len,-1,-1)+0.001*(2*T.rand((mem_len, data_width,data_width), device=self.device)-1)

        for i in range(self.mem_len):
            permutations = T.randperm(self.data_width)
            self.initial_mem_mats[i] = self.initial_mem_mats[i,permutations]
            self.initial_mem_mats[i]*=T.sign(T.rand((self.data_width,self.data_width),device=self.device)-0.5)

        self.initial_bv = T.zeros((mem_len,data_width), device=self.device)
        self.initial_bw = T.zeros((mem_len,data_width), device=self.device)
        self.initial_worth = T.zeros((mem_len,1), device=self.device)
        self.initial_write_time = T.zeros((mem_len, int(math.ceil(math.sqrt(mem_len)))), device=self.device)

        if(param_mem_mats):
            self.initial_mem_mats = T.nn.Parameter(self.initial_mem_mats)
            self.initial_bv = T.nn.Parameter(self.initial_bv)
            self.initial_bw = T.nn.Parameter(self.initial_bw)
            self.initial_worth = T.nn.Parameter(self.initial_worth)
            self.initial_write_time=T.nn.Parameter(self.initial_write_time)
        #self.initial_mem_mats[::2]=T.round(T.rand((mem_len//2,data_width,data_width),device=self.device)-0.5)





        #self.initial_mem_mats[0]=T.eye(self.data_width,device=self.device)
        #self.initial_mem_mats[1] = T.flip(T.eye(data_width,device=self.device),dims=(-1,))
        #self.initial_bv[0] = T.zeros(data_width,device=self.device)
        #self.initial_bw[0] = T.zeros(data_width,device=self.device)


        self.worth_mult=1

        self.attn_activation = (lambda x: x)#self.deNaNed(F.normalize(x**3,dim=-2)))# T.where(x.abs()-x.abs().mean(dim=-2,keepdim=True)>0, x, 0))


        #self.activation = (lambda x: x)#x/(1+T.abs(x)))
        #self.inv_activation = (lambda x: x)#T.clamp(x,min=-0.999,max=0.999)/(1-T.abs(T.clamp(x,min=-0.999,max=0.999))))

        #self.activation = (lambda x: x/(1+T.abs(x)))
        #self.inv_activation = (lambda x: T.clamp(x,min=-0.999,max=0.999)/(1-T.abs(T.clamp(x,min=-0.999,max=0.999))))

        self.lstm_activation = (lambda x: T.clamp(x,min=-0.99999,max=0.99999)/(1-T.abs(T.clamp(x,min=-0.99999,max=0.99999))))

        #self.activation = (lambda x: T.sign(x)*T.log(T.abs(x)+1))
        #self.inv_activation = (lambda x: T.sign(x)*(T.exp(T.abs(x))-1))

        #self.activation = softswish # self-normalizing (ish)

        self.activation = tapered_swish

        self.just_reset=True
        self.steps_since_reset=0




    def engram_info(self):
        bv = self.memories[self.steps_since_reset]['bv']
        bw = self.memories[self.steps_since_reset]['bw']
        worth = self.memories[self.steps_since_reset]['w']
        diag = T.diagonal(self.memories[self.steps_since_reset]['m'],dim1=-1,dim2=-2)
        antidiag = T.diagonal(T.flip(self.memories[self.steps_since_reset]['m'],dims=(-1,)),dim1=-1,dim2=-2)
        top = self.memories[self.steps_since_reset]['m'][:,:,:,0]
        bottom = self.memories[self.steps_since_reset]['m'][:,:,:,-1]
        left = self.memories[self.steps_since_reset]['m'][:,:,0,:]
        right = self.memories[self.steps_since_reset]['m'][:,:,-1,:]
        write_time = self.memories[self.steps_since_reset]['t'].to(self.device)
        current_time = self.pos_encoding(self.steps_since_reset).unsqueeze(0).unsqueeze(-2).expand(bv.size(0), bv.size(1),-1).to(self.device)

        atten_dat_size= self.data_width*9+1 + 2*int(math.ceil(math.sqrt(self.mem_len)))
        zero_padding = T.zeros((self.memories[self.steps_since_reset]['m'].size(0), self.mem_len, math.ceil(atten_dat_size/self.num_attn_heads)*self.num_attn_heads - atten_dat_size), device=self.device)


        return T.cat((bv, bw, worth, diag, antidiag, top, bottom, left, right, write_time, current_time, zero_padding), dim=-1)


    def reset(self, x):
        self.memories=[{
                'm':self.initial_mem_mats.unsqueeze(0).expand(x.size(0),-1,-1,-1).clone(),
                'bv':self.initial_bv.unsqueeze(0).expand(x.size(0),-1,-1).clone(),
                'bw':self.initial_bw.unsqueeze(0).expand(x.size(0),-1,-1).clone(),
                'w':self.initial_worth.unsqueeze(0).expand(x.size(0),-1,-1).clone(),
                't':self.initial_write_time.unsqueeze(0).expand(x.size(0),-1,-1).clone()
                }]
        temp = {
            'm':T.empty((x.size(0), self.mem_len, self.data_width, self.data_width), device=self.device),
            'bv':T.empty((x.size(0), self.mem_len, self.data_width), device=self.device),
            'bw':T.empty((x.size(0), self.mem_len, self.data_width), device=self.device),
            'w':T.empty((x.size(0), self.mem_len, 1), device=self.device),
            't':T.empty((x.size(0), self.mem_len, int(math.ceil(math.sqrt(self.mem_len)))), device=self.device)
            }
        self.memories= self.memories*2 + [temp]*(x.size(-2)+1)
        self.steps_since_reset=0
        self.hx_cx = (T.zeros((3,x.size(0),self.data_width*2*4), device=self.device), T.zeros((3,x.size(0),self.data_width*2*4), device=self.device))

    def deNaNed(self, x):
        if(isinstance(x, dict)):
            for key in x.keys():

                x[key] = T.where(x[key].isnan(), T.rand(x[key].size(),device=self.device)*0.02-0.01, x[key])
                x[key] = T.where(x[key].isposinf(), 1_000_000, x[key])
                x[key] = T.where(x[key].isneginf(), -1_000_000, x[key])
            return x
        else:
            x = T.where(x.isnan(), T.rand(x.size(),device=self.device)*0.02-0.01, x)
            x = T.where(x.isposinf(), 1_000_000, x)
            x = T.where(x.isneginf(), -1_000_000, x)
            return x

    # adapted from https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code
    @T.no_grad()
    def pos_encoding(self, index):
        data_width = int(math.ceil(math.sqrt(self.mem_len)))
        mult = 2**(T.arange(data_width//2, device=self.device)-2)
        sins = T.sign(T.sin(index * mult))
        coss = T.sign(T.cos(index * mult))

        return T.cat((sins,coss),dim=-1)



    def sparsity(self, x):  ## uses the Hoyer measure. Gini would be invariant under cloning, but Gini requires
                            ## that the data is sorted.
        return ((self.data_width**0.5) - (T.sum(x,dim=-1)/T.sum(x**2,dim=-1 )))*(((self.data_width**0.5)-1)**-1)


    def forward(self,x, correct=None, reset=True):
        if(reset):
            self.reset(x)

        if correct!=None:
                loops = x.size(-2)
                results = T.empty(x.size(), device=self.device)
                routing_losses = [T.empty(1, device=self.device)]*loops

                for i in range(loops):
                    self.steps_since_reset+=1
                    ## changes to main memory are returned as a tuple of (engrams, memory_mask).
                    ## Engrams & memory mask have the same length as memory, and memory_mask
                    ## has masks for each key/attribute of the engram. Once all changes are decided,
                    ## the memory masks are multiplied by the engram, summed together,  and then
                    ## LERP'ed into the next memory step.

                    result, routing_loss, read_mem = self.read(x[:,i], correct[:,i])


                    if(self.autoscale): ## rescale to size of correct output
                      shift = T.mean(correct[:,i], dim=-1, keepdim=True) - T.mean(result, dim=-1, keepdim=True)
                      scale = T.amax(correct[:,i],dim=-1, keepdim=True) - T.amin(correct[:,i],dim=-1, keepdim=True)
                      result = F.normalize(result-shift,dim=-1, p=float('inf'))*scale #result * (T.linalg.vector_norm(correct[:,i], ord=1)/T.linalg.vector_norm(result, ord=1))

                    results[:,i]=result
                    routing_losses[i]+=routing_loss



                    consolidate_mem = self.consolidate(5,2) ## merge some memories together.

                    inaccuracy = T.mean(F.mse_loss(result, correct[:,i], reduction='none'), dim=-1,keepdim=True) ## TODO: add hook

                    memorize_mem = self.memorize(x[:,i],  correct[:,i], log_inaccuracy=T.log(inaccuracy+1e-6))


                    new_mem = {} ## Note: might need to make tensordict.

                    for key in self.memories[self.steps_since_reset].keys():
                        mask_sum = T.clamp(read_mem[1][key] + consolidate_mem[1][key] + memorize_mem[1][key],min=1e-6)
                        #if(key=='m'):
                            #print(mask_sum[0,:,0,0])


                        new_mem[key] = self.deNaNed((read_mem[0][key] * (read_mem[1][key]/mask_sum)) + \
                                                    (consolidate_mem[0][key] * (consolidate_mem[1][key]/mask_sum)) + \
                                                    (memorize_mem[0][key] * (memorize_mem[1][key]/mask_sum)) + \
                                                    (self.memories[self.steps_since_reset][key] * T.clamp(1-mask_sum, min=0))
                                                    )
                    #new_mem = tensordict.TensorDict(new_mem, batch_size=self.memories[self.steps_since_reset].batch_size)
                    #print((new_mem['m'] - self.memories[0]['m'])[0,:,0,0])
                    #print(new_mem['m'][0,:,0,0])
                    self.memories[self.steps_since_reset+1]=new_mem

                inaccuracies = T.sum(F.mse_loss(results, correct, reduction='none'), dim=-1) ## TODO: add hook so the inaccuracies can accurately represent the error of any supermodels.


                est_worth, self.hx_cx = self.worth_model(T.cat((x,correct), dim=-1), self.hx_cx)
                estimated_worth = T.sum(self.lstm_activation(est_worth), dim=-1)
                loss_estimated_worth = F.mse_loss(T.log(inaccuracies+1e-6), estimated_worth)


                routing_losses = T.mean(T.stack(routing_losses, dim=0))

                return self.deNaNed(results), routing_losses, loss_estimated_worth

        else:
            self.input=x
            self.step_index=0
            loops = x.size(-2)
            results = T.empty(x.size(), device=self.device)
            lru_hidden=[T.empty((x.size(0), 1, self.data_width*2), device=self.device)]*x.size(-2)
            for i in range(loops):




                self.steps_since_reset+=1

                result,log_inaccuracy, read_mem = self.read(x[:,i])
                if(self.autoscale): ## rescale to size of correct output
                      shift = T.mean(x[:,i], dim=-1, keepdim=True) - T.mean(result, dim=-1, keepdim=True)
                      scale = T.amax(x[:,i],dim=-1, keepdim=True) - T.amin(x[:,i],dim=-1, keepdim=True)
                      result = F.normalize(result-shift,dim=-1, p=float('inf'))*scale #result * (T.linalg.vector_norm(correct[:,i], ord=1)/T.linalg.vector_norm(result, ord=1))

                results[:,i]=result
                consolidate_mem = self.consolidate(5,2)

                memorize_mem = self.memorize(x[:,i], result, log_inaccuracy = log_inaccuracy)

                new_mem={}

                for key in self.memories[self.steps_since_reset].keys():
                    mask_sum = T.clamp(read_mem[1][key] + consolidate_mem[1][key] + memorize_mem[1][key], min=1e-6)

                    new_mem[key] = self.deNaNed((read_mem[0][key] * (read_mem[1][key]/mask_sum)) + \
                                                (consolidate_mem[0][key] * (consolidate_mem[1][key]/mask_sum)) + \
                                                (memorize_mem[0][key] * (memorize_mem[1][key]/mask_sum)) + \
                                                (self.memories[self.steps_since_reset][key] * T.clamp(1-mask_sum, min=0))
                                                )
                #new_mem = tensordict.TensorDict(new_mem, batch_size=self.memories[self.steps_since_reset].batch_size)
                self.memories[self.steps_since_reset+1]=new_mem

            return results






    @T.inference_mode()
    def mitm_route(self, x, output):

        # y_{i} = g((x - bv_{i}) @ m_{i}) + bw_{i}
        # y = sum( y_{i} * a_{i} )

        forward_vec = x.detach()
        backward_vec = output.detach().unsqueeze(-2)
        routing = [T.zeros((x.size(0), self.mem_len,1), device=self.device) for i in range(self.max_steps)]

        worth = T.zeros((x.size(0), self.mem_len,1), device=self.device)


        ## forward half is neural router. backward half is perfect routing.
        ## backwards goal = lerp(backward_vector, forward_vector, 1/(max_steps-step)) <-- this means backwards routing will move the same distance each step.
        for i in range(self.max_steps):
            if(i%2==1):
                #backward_vec_candidates =  T.linalg.solve(self.memories[-1]['m']+T.eye(self.data_width,device=self.device) * 1e-6, self.inv_activation(backward_vec-self.memories[-1]['bw']).unsqueeze(-2), left=False).squeeze(-2)

                lerp_forward = T.lerp(backward_vec.squeeze(-2), forward_vec, 1/(self.max_steps-i))


                matrix_results = self.activation(T.matmul((lerp_forward.unsqueeze(-2).expand(-1,self.mem_len,-1)-self.memories[self.steps_since_reset]['bv']).unsqueeze(-2), self.memories[self.steps_since_reset]['m'])).squeeze(-2) + self.memories[self.steps_since_reset]['bw'] + lerp_forward.unsqueeze(-2)
                backwards_route =T.linalg.lstsq(matrix_results.mT, backward_vec.mT).solution


                ## Theory: A high sparsity in the backwards route indicates a clear solution to the problem.
                ## If there is a clear solution to the problem, the number of steps needed to complete the solution
                ## should be low, so therefore be stopped early to prevent data degradation.

                #print(self.sparsity(backwards_route[:,:,0])[0])

                ## This isn't strictly neccessary, but it pushes the router to activate when forward vec is aprox. equal to bv

                softsign =(lambda x_: x_/(1+T.abs(x_)))
                w = ((F.cosine_similarity(forward_vec.unsqueeze(-2), self.memories[self.steps_since_reset]['bv'], dim=-1).unsqueeze(-1) + \
                      F.cosine_similarity(self.pos_encoding(self.steps_since_reset).unsqueeze(0).unsqueeze(0), self.memories[self.steps_since_reset]['t'], dim=-1).unsqueeze(-1)) + \
                      softsign(self.memories[self.steps_since_reset]['w']))
                #w=T.where(backwards_route.abs()>0.5, 1,-backwards_route)
                offset = (T.eye(self.mem_len,device=self.device).unsqueeze(0)-T.linalg.lstsq(matrix_results.mT, matrix_results.mT).solution)@w
                backwards_route +=offset

                #print("offset+route",self.sparsity(backwards_route[:,:,0])[0])
                #backwards_route *=0
                #backwards_route[:,1]=1
                routing[-i//2]=backwards_route

                '''
                matrix_sum = T.sum(backwards_route.unsqueeze(-1)*self.memories[-1]['m'], dim=-3)
                bv_sum = T.sum(backwards_route*self.memories[-1]['bv'],dim=-2,keepdim=True)
                bw_sum = T.sum(backwards_route*self.memories[-1]['bw'],dim=-2,keepdim=True)
                '''
                #matrix_results = self.activation(T.matmul((lerp_forward.unsqueeze(-2).expand(-1,self.mem_len,-1)-self.memories[self.steps_since_reset]['bv']).unsqueeze(-2), self.memories[self.steps_since_reset]['m'])).squeeze(-2) + self.memories[self.steps_since_reset]['bw']
                #print( (T.sum(backwards_route*matrix_results,dim=-2) - backward_vec[:,0])[0])

                backward_vec = lerp_forward.unsqueeze(-2)#F.normalize(T.sum(backward_vec_candidates*backwards_route, dim=-2, keepdim=True), dim=(-1,), p=float('inf'))
            else:
                attention_dat = T.cat((self.engram_info(),forward_vec.unsqueeze(-2).expand(-1,self.mem_len,-1)),dim=-1)
                routing[i//2]= self.attn_activation(T.sum(self.attention_model(attention_dat), dim=-1, keepdim=True))
                matrix_results = self.activation(T.matmul((forward_vec.unsqueeze(-2).expand(-1,self.mem_len,-1)-self.memories[self.steps_since_reset]['bv']).unsqueeze(-2), self.memories[self.steps_since_reset]['m'])).squeeze(-2) + self.memories[self.steps_since_reset]['bw']
                #matrix_sum = T.sum(routing[i//2].unsqueeze(-1)*self.memories[-1]['m'], dim=-3)
                #bv_sum = T.sum(routing[i//2]*self.memories[-1]['bv'],dim=-2,keepdim=True)
                #bw_sum = T.sum(routing[i//2]*self.memories[-1]['bw'],dim=-2,keepdim=True)
                ## I'd like to use residual connections, but they don't work with the backward routing.
                forward_vec = T.sum(matrix_results*routing[i//2], dim=-2) + forward_vec
                #forward_vec = F.normalize(self.activation(T.sum(T.bmm((forward_vec.unsqueeze(-2) -bv_sum), matrix_sum)) ) + bw_sum, dim=(-1,), p=float('inf')).squeeze(-2)




            #print("Backward", backward_vec[0])
        #self.debug_vectors[:,-1]= output
        return T.stack(routing, dim=-3)








    def read(self, x, correct=None):


        #self.memories[-1]['bv'] = self.memories[-1]['bv'] + T.rand(self.memories[-1]['bv'].size(), device=self.device)
        #print(self.memories[-1]['w'][0])

        result = [x]*(self.max_steps+1)

        # y_{i} = g((x - bv_{i}) @ m_{i}) + bw_{i}
        # y = sum( y_{i} * a_{i} )



        if(correct!=None):
            routing = self.mitm_route(x, correct).clone()
            routing_loss=T.zeros(1,device=self.device)
            for i in range(self.max_steps):

                matrix_results = self.activation(T.matmul((result[i].unsqueeze(-2).expand(-1,self.mem_len,-1)-self.memories[self.steps_since_reset]['bv']).unsqueeze(-2), self.memories[self.steps_since_reset]['m'])).squeeze(-2) + self.memories[self.steps_since_reset]['bw']

                result[i+1] = T.sum(routing[:,i]*matrix_results,dim=-2) + result[i]


                if self.training and i>=self.max_steps//2:
                    attention_dat = T.cat((self.engram_info(),result[i].unsqueeze(-2).expand(-1,self.mem_len,-1)),dim=-1)
                    routing_loss += F.mse_loss(routing[:,i],self.attn_activation(T.sum(self.attention_model(attention_dat),dim=-1)).unsqueeze(-1))


            inaccuracy = T.sum(F.mse_loss(result[-1],correct, reduction='none'), dim=-1, keepdim=True)
            d_worth = T.sum(routing, dim=1)*T.log(inaccuracy+1e-6).unsqueeze(-2).expand(-1,self.mem_len,-1)/self.max_steps

            read_mem_vals ={
                'm':0,
                'bv':0,
                'bw':0,
                'w':self.memories[self.steps_since_reset]['w'] - self.worth_mult*d_worth,
                't':0
                }

            read_mem_mask = {'m':0,'bv':0,'bw':0,'w':1,'t':0}



            return result[-1], routing_loss, (read_mem_vals, read_mem_mask)



        else:
            routing = [T.empty((x.size(0), self.mem_len,1))]*self.max_steps
            for i in range(self.max_steps):
                attention_dat = T.cat((self.engram_info(),result[i].unsqueeze(-2).expand(-1,self.mem_len,-1)),dim=-1)
                routing[i]=self.attn_activation(T.sum(self.attention_model(attention_dat),dim=-1)).unsqueeze(-1)

                matrix_results = self.activation(T.matmul((result[i].unsqueeze(-2).expand(-1,self.mem_len,-1)-self.memories[self.steps_since_reset]['bv']).unsqueeze(-2), self.memories[self.steps_since_reset]['m'])).squeeze(-2) + self.memories[self.steps_since_reset]['bw']
                result[i+1] = T.sum(routing[i]*matrix_results,dim=-2) + result[i]


            ## Unfortunately, there's no pretty way to incrementally predict the worth of each output using a transformer.
            ## Ideally, I would cache the internal state resulting from each step, but pytorch doesn't allow that.
            ## torchtune does, but that's for inference, and I'm not sure it's applicable to training.




            worth_est, self.hx_cx = self.worth_model(T.cat((x,result[-1]), dim=-1).unsqueeze(-2).detach(), self.hx_cx)
            worth_est = T.sum(self.lstm_activation(worth_est),dim=-1)
            d_worth =T.sum(T.stack(routing,dim=1), dim=1)*worth_est.unsqueeze(-2).expand(-1,self.mem_len,-1)/self.max_steps

            read_mem_vals={
                    'm':0,
                    'bv':0,
                    'bw':0,
                    'w': self.memories[self.steps_since_reset]['w']-self.worth_mult*d_worth,
                    't':0
                    }

            read_mem_mask = {'m':0,'bv':0,'bw':0,'w':1,'t':0}

        return result[-1], worth_est, (read_mem_vals, read_mem_mask)



    def memorize(self, x, y, log_inaccuracy=None): ## overly simplistic. Memorry consolidation is done elsewhere.

        x = F.normalize(x,dim=-1)#T.where(T.sum(T.abs(x),dim=-1, keepdim=True)>self.data_width, F.normalize(x,dim=-1), x)
        y = F.normalize(y,dim=-1)#T.where(T.sum(T.abs(y),dim=-1, keepdim=True)>self.data_width, F.normalize(y,dim=-1), y)


        new_bv = x
        new_bw = y



        #min_worth_idx = (T.rand(x.size(0),device=self.device)*self.mem_len).int()
        min_worth_idx = T.min(self.memories[self.steps_since_reset]['w'].squeeze(-1),dim=-1)[1]

        idx = T.arange(self.memories[self.steps_since_reset]['m'].size(0), device=self.device)

        memorize_mem_vals ={
                'm':T.eye(self.data_width, device=self.device),
                'bv':new_bv.unsqueeze(-2),
                'bw':new_bw.unsqueeze(-2),
                'w':-log_inaccuracy.unsqueeze(-2) if log_inaccuracy!=None else T.median(self.memories[self.steps_since_reset]['w'].squeeze(-1),dim=-1)[0].unsqueeze(-1).unsqueeze(-2),
                't':self.pos_encoding( self.steps_since_reset).to(self.device).unsqueeze(-2).expand(x.size(0),-1,-1)
                }


        mask = T.zeros((x.size(0), self.mem_len,1), device=self.device)
        mask[idx, min_worth_idx] = T.ones([1], device=self.device)

        memorize_mem_mask={'m':mask.unsqueeze(-1),'bv':mask,'bw':mask,'w':mask,'t':mask}

        return (memorize_mem_vals, memorize_mem_mask)

    def consolidate(self, num_k=5, loops=1):

        # finds the index of the engram with the most similarities.
        delta_b = (self.memories[self.steps_since_reset]['bw']-self.memories[self.steps_since_reset]['bv'])* T.sign(T.sum(T.abs(self.memories[self.steps_since_reset]['m']), dim=(-1,-2))).unsqueeze(-1)
        cos_sim = F.cosine_similarity(delta_b.unsqueeze(-2), delta_b.unsqueeze(-2).transpose(-2,-3),dim=-1)
        cos_sim += T.rand(cos_sim.size(),device=self.device)*0.002-0.001 ## inject a little bit of randomness.

        cos_sim = T.where(cos_sim.bool(), cos_sim,-1)-2*T.eye(self.mem_len, device=self.device).unsqueeze(0)

       #consolidate_mem_vals = map( lambda x: x.clone().detach(), self.memories[self.steps_since_reset])
        consolidate_mem_vals={
            'm':self.memories[self.steps_since_reset]['m'].clone().detach(),
            'bv':self.memories[self.steps_since_reset]['bv'].clone().detach(),
            'bw':self.memories[self.steps_since_reset]['bw'].clone().detach(),
            'w':0,#self.memories[self.steps_since_reset]['w'].clone().detach(),
            't':self.memories[self.steps_since_reset]['t'].clone().detach()
        }
        consolidate_mem_mask = {
            'm':T.zeros((self.memories[self.steps_since_reset]['m'].size(0), self.mem_len,1,1), device=self.device),
            'bv':T.zeros((self.memories[self.steps_since_reset]['bv'].size(0), self.mem_len,1), device=self.device),
            'bw':T.zeros((self.memories[self.steps_since_reset]['bw'].size(0), self.mem_len,1), device=self.device),
            'w':0,
            't':T.zeros((self.memories[self.steps_since_reset]['t'].size(0), self.mem_len,1), device=self.device)
            }


        for i in range(loops):
            most_similarities = T.max(T.max(F.relu(cos_sim),dim=-1)[0]/(1-T.mean(F.relu(cos_sim)+0.0001,dim=-1)), dim=-1)[1]
            idx = T.arange(self.memories[self.steps_since_reset]['m'].size(0), device=self.device)
            cos_sim += T.eye(self.mem_len, device=self.device).unsqueeze(0)*2 ## We want the index of the most similarities to be in the top k.
            topk = T.topk(cos_sim[idx,most_similarities,:],num_k,dim=-1).indices

            # generate new engram
            bv_diff = self.deNaNed(self.memories[self.steps_since_reset]['bv'][:,topk][:,0]-T.mean(self.memories[self.steps_since_reset]['bv'][:,topk],dim=-2,keepdim=True)[:,0])
            bw_diff = self.deNaNed(self.memories[self.steps_since_reset]['bw'][:,topk][:,0]-T.mean(self.memories[self.steps_since_reset]['bw'][:,topk],dim=-2,keepdim=True)[:,0])

            # lazy way to prevent non-full rank issues
            bv_diff += T.rand(bv_diff.size(), device=self.device) * 0.0001
            bw_diff += T.rand(bw_diff.size(), device=self.device) * 0.0001

            try:
                generated_matrix = T.linalg.lstsq(bv_diff, bw_diff).solution
            except Exception as e:
                print(e)
                print(bv_diff, bw_diff)
                exit()
            new_bv = T.mean(self.memories[self.steps_since_reset]['bv'][:,topk][:,0],dim=-2)
            new_bw = T.mean(self.memories[self.steps_since_reset]['bw'][:,topk][:,0],dim=-2)



            # remove nans
            generated_matrix = T.where(generated_matrix.isnan(), 0.0001, generated_matrix)
            new_bv = T.where(new_bv.isnan(), 0, new_bv)
            new_bw = T.where(new_bw.isnan(), 0, new_bw)


            # replace old engram

            consolidate_mem_vals['m'][idx,most_similarities] = generated_matrix
            consolidate_mem_vals['bv'][idx,most_similarities] = new_bv
            consolidate_mem_vals['bw'][idx,most_similarities] = new_bw
            consolidate_mem_vals['t'][idx,most_similarities] = self.pos_encoding( self.steps_since_reset).to(self.device)

            consolidate_mem_mask['m'][idx, most_similarities]+=1
            consolidate_mem_mask['bv'][idx, most_similarities]+=1
            consolidate_mem_mask['bw'][idx, most_similarities]+=1
            consolidate_mem_mask['t'][idx, most_similarities]+=1

            # set the similarity of previous consolidated engram to -1.
            cos_sim[idx,most_similarities]*=0
            cos_sim[idx,most_similarities]-=1
            cos_sim = T.clamp(cos_sim, min=-1, max=1)

        return (consolidate_mem_vals, consolidate_mem_mask)

