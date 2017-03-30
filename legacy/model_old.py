#import objgraph
#import memory_profiler
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from Generator.utils import Loader,Evaluate,TrainIterator,DevIterator,log

#theano.config.compute_test_value = 'warn'
#theano.config.exception_verbosity = 'high'
theano.config.mode="FAST_COMPILE"
theano.config.scan.allow_gc=False
word_emb=None
vocab=None
oov_id=None
numWords = 124164
unupdate_length = None
update_length = None
layers = {'lstm': ('param_init_lstm','lstm_layer'),
          'match_lstm':('param_init_match_lstm','match_lstm_layer'),
          'pointer':('param_init_pointer', 'pointer_layer'),
            }
            
not_train_params=["Unupdate"]
not_save_params=["Unupdate"]

extra_params={}

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.items():
        if kk in tparams: 
            tparams[kk].set_value(vv)
        if kk in extra_params: 
            extra_params[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params
    
def save_params(tparams):
    new_params = OrderedDict()
    for kk, vv in tparams.items():
        if kk not in not_save_params:
            new_params[kk] = vv.get_value()
    return new_params

# load parameters
def load_params(path, tparams):
    params = np.load(path)
    for kk, vv in params.items():
        if kk in tparams: 
            tparams[kk].set_value(vv)
        if kk in extra_params: 
            extra_params[kk].set_value(vv)
        
# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.items()]

def init_word_emb_t():
    global word_emb, vocab
    word_emb, vocab=Loader.load_word_emb_for_train()

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

    
def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))
    
    
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')
    

def norm_weight(nin, nout=None, scale=0.01, ortho=False):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')
    
    
def init_emb(options,params):
    embedding=word_emb["emb"]
    params['Unupdate'] = embedding.astype('float32')
    global update_length
    update_length = numWords - embedding.shape[0]
    print (update_length)
    global unupdate_length
    unupdate_length = embedding.shape[0]
    print (unupdate_length)
    oov=np.zeros((update_length,options["dim_word"]),dtype='float32')
    params['update'] = oov.astype('float32')

    # new_emb=np.concatenate((embedding,oov),axis=0)
    # global oov_id
    # oov_id=np.shape(new_emb)[0]-1
    # params['Wemb'] = new_emb.astype('float32')

    return params
    
    
def _p(pp, name):
    return '%s_%s' % (pp, name)
    
def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise,
        state_before * trng.binomial(state_before.shape, p = 0.6, n = 1, dtype = state_before.dtype),
        state_before)
    return proj

def param_init_lstm(options, params, prefix, nin, dim):
        
    W = np.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                        #    norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([norm_weight(dim),
                           norm_weight(dim),
                        #    norm_weight(dim),
                           norm_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
        
    b = np.zeros((3 * dim,))
    params[_p(prefix, 'b')] = b.astype('float32')
    
    return params

def lstm_layer(tparams, state_below, options, prefix='lstm'):
    nsteps = state_below.shape[0]
    
    def _slice(_x, n, dim):
        return _x[n * dim:(n + 1) * dim]
    def _step(x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        
        i = T.nnet.sigmoid(_slice(preact, 0, options['dim']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim']))
        # o = T.nnet.sigmoid(_slice(preact, 2, options['dim']))
        c = T.tanh(_slice(preact, 2, options['dim']))
        
        c = f * c_ + i * c
        
        h = T.tanh(c)
        
        return h, c
        
    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
                   
    dim = options['dim']
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[T.alloc(0.,dim),
                                              T.alloc(0.,dim)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]
    
    
def param_init_match_lstm(options, params, prefix, nin, dim):
    Wq = norm_weight(nin)
    params[_p(prefix, 'Wq')] = Wq
    
    Wp = norm_weight(nin)
    params[_p(prefix, 'Wp')] = Wp
    
    Wr = norm_weight(nin)
    params[_p(prefix, 'Wr')] = Wr
    
    bp = np.zeros((dim,))
    params[_p(prefix, 'bp')] = bp.astype('float32')
    
    w_att = norm_weight(dim,1)
    params[_p(prefix, 'w_att')] = w_att
    
    b_att = np.zeros((1,))
    params[_p(prefix, 'b_att')] = b_att.astype('float32')
    
    W = np.concatenate([norm_weight(2 * nin,dim),
                           norm_weight(2 * nin,dim),
                           norm_weight(2 * nin,dim),
                           norm_weight(2 * nin,dim)], axis=1)
    
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([norm_weight(dim),
                           norm_weight(dim),
                           norm_weight(dim),
                           norm_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
        
    b = np.zeros((4 * dim,))
    params[_p(prefix, 'b')] = b.astype('float32')
    
    return params

def match_lstm_layer(tparams, Hp, Hq, options, prefix='match_lstm'):
    nsteps = Hp.shape[0]
    
    def _slice(_x, n, dim):
        return _x[n * dim:(n + 1) * dim]
        
    def _step(x_, h_, c_, hq):
        
        g_r = T.dot(x_, tparams[_p(prefix, 'Wp')]) + T.dot(h_, tparams[_p(prefix, 'Wr')]) + tparams[_p(prefix, 'bp')]
        
        g_l= T.dot(hq, tparams[_p(prefix, 'Wq')])    
        
        G = T.tanh(g_l + g_r)
        
        a = T.dot(G, tparams[_p(prefix, 'w_att')]) + tparams[_p(prefix, 'b_att')]
        a = T.nnet.softmax(a.flatten())
        
        z = T.concatenate([x_, T.dot(a.flatten(), hq)],axis = 0)
        
        
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += T.dot(z, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        
        i = T.nnet.sigmoid(_slice(preact, 0, options['dim']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['dim']))
        c = T.tanh(_slice(preact, 3, options['dim']))
        
        c = f * c_ + i * c
        
        h = o * T.tanh(c)
        
        return h, c
        
    #state_below = (T.dot(state_below, tparams[_p(prefix, 'Wp')]) +
     #              tparams[_p(prefix, 'bp')])
                   
    
    
    dim = options['dim']
    rval, updates = theano.scan(_step,
                                sequences=[Hp],
                                outputs_info=[T.alloc(0.,dim),
                                              T.alloc(0.,dim)],
                                non_sequences=[Hq],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


def param_init_pointer(options, params, prefix, nin, dim):
    V = norm_weight(nin,dim)
    params[_p(prefix, 'V')] = V
    
    Wa = norm_weight(dim)
    params[_p(prefix, 'Wa')] = Wa
    
    ba = np.zeros((dim,))
    params[_p(prefix, 'ba')] = ba.astype('float32')
    
    v_att = norm_weight(dim,1)
    params[_p(prefix, 'v_att')] = v_att
    
    c_att = np.zeros((1,))
    params[_p(prefix, 'c_att')] = c_att.astype('float32')
    
    W = np.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([norm_weight(dim),
                           norm_weight(dim),
                           norm_weight(dim),
                           norm_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
        
    b = np.zeros((4 * dim,))
    params[_p(prefix, 'b')] = b.astype('float32')
    
    return params


def pointer_layer(tparams, Hr, options, prefix='pointer_layer'):
    
    def _slice(_x, n, dim):
        return _x[n * dim:(n + 1) * dim]
        
    def _step(a_, h_, c_, hr):   
        f_r = T.dot(h_, tparams[_p(prefix, 'Wa')]) + tparams[_p(prefix, 'ba')]
        
        f_l=T.dot(hr, tparams[_p(prefix, 'V')])
        
        F = T.tanh(f_l+f_r)
        
        a = T.dot(F, tparams[_p(prefix, 'v_att')]) + tparams[_p(prefix, 'c_att')]

        #a = T.log(a.flatten())
        a = T.nnet.softmax(a.flatten())
        
        z = T.dot(a.flatten(), hr)
        
        
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += T.dot(z, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        
        i = T.nnet.sigmoid(_slice(preact, 0, options['dim']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['dim']))
        c = T.tanh(_slice(preact, 3, options['dim']))
        
        c = f * c_ + i * c
        
        h = o * T.tanh(c)
        
        return a.flatten(), h, c
        
    #state_below = (T.dot(state_below, tparams[_p(prefix, 'Wp')]) +
     #              tparams[_p(prefix, 'bp')])
            
    
    dim = options['dim']
    rval, updates = theano.scan(_step,
                                outputs_info=[T.alloc(0.,Hr.shape[0]), 
                                              T.alloc(0.,dim),
                                              T.alloc(0.,dim)],
                                non_sequences=[Hr],
                                name=_p(prefix, '_layers'),
                                n_steps=options["ans_len"])
    return rval[0]
    

def init_params(options):
    params = OrderedDict()
    #embedding
    params = init_emb(options,params)
    #params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    #lstm
    params = get_layer('lstm')[0](options, params,
                                        prefix='lstm_preprocess',
                                        nin=options['dim_word'],
                                        dim=options['dim'])
    # params = get_layer('lstm')[0](options, params,
    #                                     prefix='lstm_preprocess_q',
    #                                     nin=options['dim_word'],
    #                                     dim=options['dim'])
    params = get_layer('match_lstm')[0](options, params,
                                        prefix='match_lstm',
                                        nin=options['dim'],
                                        dim=options['dim'])
    
    # params = get_layer('match_lstm')[0](options, params,
    #                                     prefix='match_lstm_b',
    #                                     nin=options['dim'],
    #                                     dim=options['dim'])
    
    params = get_layer('pointer')[0](options, params,
                                        prefix='pointer_layer',
                                        nin = 2*options['dim'],
                                        dim = options['dim'])
                                        
    return params

def build_model(params, tparams, options):
    p = T.matrix('p',dtype = 'int64')
    q = T.matrix('q',dtype = 'int64')
    
    ans = T.matrix('ans', dtype = 'int64')

    use_noise = theano.shared(np.float32(0.))
    trng = RandomStreams(1234)

    unupdate_size = theano.shared(unupdate_length)
    n_psteps = p.shape[0]
    n_qsteps = q.shape[0]

    embed =  T.concatenate([tparams['Unupdate'],tparams['update']],axis = 0)
    
    # def _get_embed(p_, h_, emb1, emb2):
    #     # print (p_[0])
    #     if p_[0] - unupdate_size >= 0:
    #         h_ = emb2[p_[0]-unupdate_size]
    #     else:
    #         h_ = emb1[p_[0]]
    #     return h_
    # emb_p, update = theano.scan(_get_embed, sequences = [p], 
    #                             outputs_info = [T.alloc(0.,options['dim_word'])], 
    #                             non_sequences=[tparams['Unupdate'],tparams['update']],
    #                              n_steps = n_psteps)

    emb_p = embed[p.flatten()]
    emb_p = emb_p.reshape([n_psteps, options['dim_word']])

    # emb_q, update = theano.scan(_get_embed, sequences = [q], 
    #                             outputs_info = [T.alloc(0.,options['dim_word'])], 
    #                             non_sequences=[tparams['Unupdate'],tparams['update']],
    #                              n_steps = n_qsteps)
    emb_q = embed[q.flatten()]
    emb_q = emb_q.reshape([n_qsteps, options['dim_word']])
    
    if options['use_dropout']:
        emb_p = dropout_layer(emb_p, use_noise, trng)
        emb_q = dropout_layer(emb_q, use_noise, trng)
    #Hp
    Hp = get_layer('lstm')[1](tparams, emb_p, options, 
                            prefix = 'lstm_preprocess')
                            
    Hq = get_layer('lstm')[1](tparams, emb_q, options, 
                            prefix = 'lstm_preprocess')

    Hq = T.concatenate([Hq, T.alloc(0.,1,Hq.shape[1])],axis = 0)

    Hp_b= Hp[::-1]
                            
    #match_layer
    Hr_f = get_layer('match_lstm')[1](tparams, Hp, Hq, options,
                                prefix = 'match_lstm')    
    Hr_b = get_layer('match_lstm')[1](tparams, Hp_b, Hq, options,
                                prefix = 'match_lstm')[::-1]
    
    Hr = T.concatenate([Hr_f,Hr_b],axis = 1)
    if options['use_dropout'] and options['test']:
        Hr = dropout_layer(Hr, use_noise, trng)
    Hr = T.concatenate([Hr, T.alloc(0.,1,Hr.shape[1])],axis = 0)
    #ans_len = ans.shape[0]
    
    # answer pointer layer
    proj = get_layer('pointer')[1](tparams, Hr, options,
                                        prefix = 'pointer_layer')
    
    
    def _step(p_, a_, h_):
        h_ = -T.log(p_[a_])
        return h_
    cost_out, update = theano.scan(_step, sequences = [proj, T.concatenate([ans.flatten(),np.array([0])])], 
                                outputs_info = [T.as_tensor_variable(np.float32(0.))], 
                                 n_steps = proj.shape[0])
    cost = cost_out.sum(0)/proj.shape[0]
    
    out = proj
    
    return trng, use_noise, p, q, ans, cost, proj, out
    
def adamax(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):
    
    
    beta1=np.float32(beta1)
    beta2=np.float32(beta2)
    e=np.float32(e)

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) \
                    for k, p in tparams.items()] \
             +[theano.shared(np.zeros(1,dtype="float32")[0], name="count")]
    gsup = [(gs, gs+g) for gs, g in zip(gshared[:-1], grads)] \
         + [(gshared[-1], gshared[-1]+np.float32(1.0))]


    updates = []
    
    #updates.extend([(g ,g / gshared[-1]) for g in gshared[:-1]])

    t_prev = theano.shared(np.float32(0.),"adam_t")
    extra_params["adam_t"]=t_prev
    t = t_prev + 1.
    lr_t = lr / (np.float32(1.) - beta1**t)
    
    for p, _g in zip(tparams.values(), gshared[:-1]):
        if p.name in not_train_params:
            continue
        g=_g/gshared[-1]
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        extra_params[p.name+'_mean']=m
        extra_params[p.name+'_variance']=v

        m_t = beta1 * m + (np.float32(1.) - beta1) * g
        v_t = T.maximum(beta2 * v, T.abs_(g))
        step = lr_t * m_t / (v_t + e)

        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))
    
    updates.extend([(v, T.zeros(v.shape,dtype='float32')) for v in gshared[:-1]])  # Set accumulators to 0
    updates.append((gshared[-1],  T.zeros(1,dtype='float32')[0]))

    f_update = theano.function([lr], [], updates=updates)
    log.info("substep done")
    f_grad_acc = theano.function(inp, cost, updates=gsup)
                               #on_unused_input='ignore'

    return f_grad_acc, f_update
    

def prepare_data(seqs_p, seqs_q, seqs_ans):
    p = np.zeros((len(seqs_p),1)).astype('int64')
    q = np.zeros((len(seqs_q),1)).astype('int64')
    ans = np.zeros((len(seqs_ans),1)).astype('int64')
    
    for i in range(p.shape[0]):
        p[i][0] = seqs_p[i]
    for i in range(q.shape[0]):
        q[i][0] = seqs_q[i]
    for i in range(ans.shape[0]):
        ans[i][0] = seqs_ans[i]
    return p, q, ans
    
def prepare_eval_data(seqs_p, seqs_q):
    p = np.zeros((len(seqs_p),1)).astype('int64')
    q = np.zeros((len(seqs_q),1)).astype('int64')
    
    for i in range(p.shape[0]):
        p[i][0] = seqs_p[i]
    for i in range(q.shape[0]):
        q[i][0] = seqs_q[i]
    return p, q
    
def build_ans(seqs_p,ans):
    if ans[0]>ans[1]: 
        ans[1]=ans[0]
    anss=""    
    for i in range(ans[0],ans[1]+1):
        if i>ans[0]: anss+=" "
        anss+=seqs_p[i]
    return anss
        
def build_ans_dev(seqs_p,ans):
    if ans[0]>ans[1]: 
        ans[1]=ans[0]
    anss=""
    length = 0   
    for i in range(ans[0],ans[1]+1):
        # if i>ans[0]: anss+=" "
        # anss+=seqs_p[i][0]
        while length + 1 != seqs_p[i][1] and i != ans[0]:
            anss+=" "
            length += 1

        anss+=seqs_p[i][0]
        length = seqs_p[i][2]

    return anss
        
def get_ans_from_proj(proj):
    max_score = -9999999999
    indexs = []
    for i in range(proj.shape[1]-1):
        for j in range(0,15):
            if i + j >= proj.shape[1]-1:
                break
            score = proj[0][i] * proj[1][i+j]
            if score > max_score:
                max_score = score
                indexs = [i, i+j]
    return indexs

def train(dim_word=300,  # word vector dimensionality
          dim = 150,
          n_batch_per_epoch=50,
          max_epochs=300,
          lrate=0.001,  # learning rate
          ans_len=2,
          optimizer='adamax',
          batch_size=30,
          saveto='model_results_0.001',
          loadfrom=None,
          logfile="model_0.001.log",
          test=True,
          use_dropout=True):
          #reload_=False,
          #overwrite=False,

    np.random.seed(314)
    random.seed(314)
    
    log.init(logfile)

    model_options = locals().copy()

    log.info('loading data')
    
    init_word_emb_t()
    train_iter = TrainIterator(word_emb,vocab,batch_size)
        
    log.info('building model')
    
    params = init_params(model_options)
    
    tparams = init_tparams(params)
    
    
    if loadfrom!=None:
        log.info("load pretrained model")
        load_params(loadfrom,tparams)
        log.info("done")    
    
    trng, use_noise, p, q, ans, cost,proj,out = build_model(params,tparams, model_options)
    
    inps = [p, q, ans]
    inps_eval=[p,q]
    
    #f_cost = T.function(inps, cost)
    
    grads = T.grad(cost, wrt=itemlist(tparams))
    lr = T.scalar(name='lr', dtype='float32')
    
    log.info("finish step 1")
    
    f_grad_acc, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    
    f_evaluate=theano.function(inps_eval,out)    
    #f_grad_shared = eval(optimizer)(lr, tparams, grads, inps, cost)
    
    log.info("finish step 2")    
    
    log.info('done')
    

    log.info(unupdate_length)
    log.info(update_length)
    # gc.collect() 
    cost = 0
    for idx in range(max_epochs):
        epoch_cost=0.0
        epoch_acc=0.0
        us_start = time.time()
        
        
        log.info("training start")
            
        for j in range(n_batch_per_epoch):
            cost = 0.0
            ps,qs,anss,tps=next(train_iter)            
            
            acc=0.0
            #lrate = lrate * 0.95
            for i in range(batch_size):
                use_noise.set_value(1.)
                test = True

                p, q, ans = prepare_data(ps[i],qs[i],anss[i])
                
                e_ans=f_evaluate(p,q)

                e_ans = get_ans_from_proj(e_ans)

                e_ansss=build_ans(tps[i],e_ans)                
                # print (anss[i])
                ansss=build_ans(tps[i],anss[i])
                
                acc+=Evaluate.f1_score(e_ansss,ansss)
                
                cost += f_grad_acc(p, q, ans)
                        
            acc/=float(batch_size)
            cost/=float(batch_size)
            
            f_update(np.float32(lrate))
            
            epoch_cost+=cost
            epoch_acc+=acc
            #log.info(objgraph.show_most_common_types(limit=100))
            log.info("batch = %d cost = %f global cost = %f f1 = %f global f1 = %f"
            %(j,cost,epoch_cost/float(j+1),acc,epoch_acc/float(j+1)))
        epoch_cost/=float(n_batch_per_epoch)
        epoch_acc/=float(n_batch_per_epoch)
        log.info("training done")
        
        #Evaluate
        log.info("evaluate start")

        i_eval=0        
        dev_iter = DevIterator(word_emb,vocab)
        predictions={}
        for _idx,ps,qs,tps in dev_iter:
            test = False
            use_noise.set_value(0.)
            p,q=prepare_eval_data(ps,qs)
            #assert len(ps)==len(tps)
            ans=f_evaluate(p,q)
            ans = get_ans_from_proj(ans)
            #log.info(ans)
#            try:
            anss=build_ans(tps,ans)
#            except:
#                log.info(len(tps),len(p),ans)
#                log.info(proj)
#                raise
            predictions[_idx]=anss
            i_eval+=1
            if i_eval%1000==0: log.info(i_eval)
        log.info(i_eval)
        log.info("evaluate done")
        eval_result=Evaluate.evaluate(predictions)
        
        log.info("save")
        if os.path.isdir(saveto)==False:
            os.makedirs(saveto)
        file_name=os.path.join(saveto,"%d_%.2f%%_%.2f%%.npz"%(idx,eval_result["exact_match"]*100,eval_result["f1"]*100))
        tparamss=save_params(merge_two_dicts(tparams,extra_params))        
        np.savez(file_name,**tparamss)  
        file_name=os.path.join(saveto,"%d_%.2f%%_%.2f%%.json"%(idx,eval_result["exact_match"]*100,eval_result["f1"]*100))
        with open(file_name,"w") as f:
            json.dump(predictions,f)
        file_name=os.path.join(saveto,"%d_%.2f%%_%.2f%%.txt"%(idx,eval_result["exact_match"]*100,eval_result["f1"]*100))
        with open(file_name,"w") as f:
            json.dump(eval_result,f)        
        ud = time.time() - us_start
        log.info('epoch = %d'%(idx,))
    
        log.info('time spent = %f'%(ud,))
        
        
        log.info('global cost = %f'%(epoch_cost,))
        
        log.info("train f1 = %f"%(epoch_acc,))        
        
        log.info("dev exact match = %f"%(eval_result["exact_match"],))
        log.info("dev f1 score = %f"%(eval_result["f1"],))
        log.info("dev precision = %f"%(eval_result["precision"],))
        log.info("dev recall = %f"%(eval_result["recall"],))
        
                
                
if __name__ == '__main__':
    train()
