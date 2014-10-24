import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
import scipy as sp
from matplotlib import animation
from matplotlib.path import Path
import cPickle as cp
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('/home/eweiss/Desktop/Sum-of-Functions-Optimizer/')
from sfo import SFO

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


nx=2
nsamps=64000
#n_subfuncs=int(np.round(np.sqrt(nsamps)/10.0))
#batchsize=int(np.round(10.0*np.sqrt(nsamps)))
n_subfuncs=1
batchsize=int(nsamps/n_subfuncs)
nsteps=40
nbeta=20
#betas=(-2.0*np.ones(nsteps)).astype(np.float32)
betas=np.zeros(nbeta).astype(np.float32)
betas[0]=-0.0
beta_max=np.ones(nsteps)#*(1. - np.exp(np.log(0.6)/float(nsteps)) + 0.0001)
#beta_max[0]*=0.001
beta_max=beta_max

nhid_mu=4**2
nhid_cov=4**2
ntgates=16

n_epochs=32*48

save_forward_animation=False
save_reverse_animation=False
plot_reverse_process=True
automate_training=False

save_model_and_optimizer=False
save_fn='model_optimizer_learn_beta_18tgates_20T_noisier_bigdata.cpl'

load_model=True
load_fn='model_optimizer_learn_beta_18tgates_20T_noisier_bigdata.cpl'

kT=-np.log(0.5)*8.0*ntgates**2

xlm=3
ylm=3

mu_centers=(np.random.randn(nx, nhid_mu)*1.0).astype(np.float32)
mu_spreads=(np.zeros((nx, nhid_mu))-1.0).astype(np.float32)
mu_biases=np.zeros(nhid_mu).astype(np.float32)
mu_M=(np.random.randn(nhid_mu, ntgates*nx)*0.00001).astype(np.float32)
mu_b=np.zeros((ntgates, nx)).astype(np.float32)

cov_centers=(np.random.randn(nx, nhid_cov)*1.0).astype(np.float32)
cov_spreads=(np.zeros((nx, nhid_cov))-1.0).astype(np.float32)
cov_biases=np.zeros(nhid_cov).astype(np.float32)
cov_M=(np.random.randn(nhid_cov, ntgates)*0.00001).astype(np.float32)
cov_b=np.zeros(ntgates).astype(np.float32)

theano_rng = RandomStreams()



def whiten(x):
	mu=np.mean(x,axis=0)
	x=x-mu
	cov=np.cov(x.T)
	cov_inv=np.linalg.inv(cov)
	cov_inv_sqrt=sp.linalg.sqrtm(cov_inv)
	out=np.dot(x,cov_inv_sqrt)
	return out


def compute_betas(betaparams, t):
	ts=T.extra_ops.repeat(t.dimshuffle(0,'x'),betaparams.shape[0],axis=1)
	pows=T.extra_ops.repeat(T.arange(betaparams.shape[0]).dimshuffle('x',0),t.shape[0],axis=0)
	out=T.sum((ts**pows)*betaparams.dimshuffle('x',0),axis=1)
	return beta_max*T.nnet.sigmoid(out)


def compute_betas_numpy(betaparams):
	t=np.arange(nsteps)/float(nsteps)
	ts=np.repeat(t.reshape((nsteps,1)),betaparams.shape[0],axis=1)
	pows=np.repeat(np.arange(betaparams.shape[0]).reshape((1,nbeta)),t.shape[0],axis=0)
	out=np.sum((ts**pows)*betaparams.reshape((1,nbeta)),axis=1)
	return (1.0/(1.0+np.exp(-out)))*beta_max


def compute_f_mu(x, t, params):
	[centers, spreads, biases, M, b]=params
	diffs=x.dimshuffle(0,1,2,'x')-centers.dimshuffle('x','x',0,1)
	scaled_diffs=(diffs**2)*T.exp(spreads).dimshuffle('x','x',0,1)
	exp_terms=T.sum(scaled_diffs,axis=2)+biases.dimshuffle('x','x',0)*0.0
	h=T.exp(-exp_terms)
	sumact=T.sum(h,axis=2)
	#Normalization
	hnorm=h/sumact.dimshuffle(0,1,'x')
	z=T.dot(hnorm,M)
	z=T.reshape(z,(t.shape[0],t.shape[1],ntgates,nx))+b.dimshuffle('x','x',0,1) #nt by nb by ntgates by nx
	#z=z+T.reshape(x,(t.shape[0],t.shape[1],1,nx))
	
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,muWT)+mubT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
	tgating=T.reshape(tgating,(t.shape[0],t.shape[1],ntgates,1))
	
	mult=z*tgating
	
	out=T.sum(mult,axis=2)
	
	#out=out+x
	
	return T.cast(out,'float32')


def compute_f_cov(x, t, params):
	[centers, spreads, biases, M, b]=params
	diffs=x.dimshuffle(0,1,2,'x')-centers.dimshuffle('x','x',0,1)
	scaled_diffs=(diffs**2)*T.exp(spreads).dimshuffle('x','x',0,1)
	exp_terms=T.sum(scaled_diffs,axis=2)+biases.dimshuffle('x','x',0)*0.0
	h=T.exp(-exp_terms)
	sumact=T.sum(h,axis=2)
	#Normalization
	hnorm=h/sumact.dimshuffle(0,1,'x')
	z=T.dot(hnorm,M)
	z=T.reshape(z,(t.shape[0],t.shape[1],ntgates))+b.dimshuffle('x','x',0) #nt by nb by ntgates
	z=T.exp(z)
	
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,covWT)+covbT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
	tgating=T.reshape(tgating,(t.shape[0],t.shape[1],ntgates))
	
	mult=z*tgating
	
	out=T.sum(mult,axis=2)
	
	return T.cast(out,'float32')


def compute_mu_field(x, t, params):
	t0=T.cast(T.arange(x.shape[0])*0.0+t, 'float32')
	t1=T.reshape(t0,(x.shape[0],1,1))
	t2=T.extra_ops.repeat(t1,x.shape[1],axis=1)
	[centers, spreads, biases, M, b]=params
	diffs=x.dimshuffle(0,1,2,'x')-centers.dimshuffle('x','x',0,1)
	scaled_diffs=(diffs**2)*T.exp(spreads).dimshuffle('x','x',0,1)
	exp_terms=T.sum(scaled_diffs,axis=2)+biases.dimshuffle('x','x',0)*0.0
	h=T.exp(-exp_terms)
	sumact=T.sum(h,axis=2)
	#Normalization
	hnorm=h/sumact.dimshuffle(0,1,'x')
	z=T.dot(hnorm,M)
	z=T.reshape(z,(x.shape[0],x.shape[1],ntgates,nx))+b.dimshuffle('x','x',0,1) #nt by nb by ntgates by nx
	#z=z+T.reshape(x,(t.shape[0],t.shape[1],1,nx))
	
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,muWT)+mubT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t2)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t2.shape[0], t2.shape[1], 1))
	tgating=T.reshape(tgating,(t2.shape[0],t2.shape[1],ntgates,1))
	
	mult=z*tgating
	
	out=T.sum(mult,axis=2)
	
	return T.cast(out,'float32')


def compute_cov_field(x, t, params):
	t0=T.cast(T.arange(x.shape[0])*0.0+t, 'float32')
	t1=T.reshape(t0,(x.shape[0],1,1))
	t2=T.extra_ops.repeat(t1,x.shape[1],axis=1)
	[centers, spreads, biases, M, b]=params
	diffs=x.dimshuffle(0,1,2,'x')-centers.dimshuffle('x','x',0,1)
	scaled_diffs=(diffs**2)*T.exp(spreads).dimshuffle('x','x',0,1)
	exp_terms=T.sum(scaled_diffs,axis=2)+biases.dimshuffle('x','x',0)*0.0
	h=T.exp(-exp_terms)
	sumact=T.sum(h,axis=2)
	#Normalization
	hnorm=h/sumact.dimshuffle(0,1,'x')
	z=T.dot(hnorm,M)
	z=T.reshape(z,(x.shape[0],x.shape[1],ntgates))+b.dimshuffle('x','x',0) #nt by nb by ntgates by 1
	#z=z+T.reshape(x,(t.shape[0],t.shape[1],1,nx))
	z=T.exp(z)
	
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,muWT)+mubT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t2)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t2.shape[0], t2.shape[1], 1))
	tgating=T.reshape(tgating,(t2.shape[0],t2.shape[1],ntgates))
	
	mult=z*tgating
	
	out=T.sum(mult,axis=2)
	
	return T.cast(out,'float32')


def forward_step(b, x):
	
	samps=theano_rng.normal(size=x.shape)*T.sqrt(b)
	means=x*T.sqrt(1.0-b)
	return T.cast(means+samps,'float32')


def compute_forward_trajectory(x0,beta_params):
	
	tpoints=T.cast(T.arange(nsteps),'float32')/T.cast(nsteps,'float32')
	betas=compute_betas(beta_params,tpoints)
	
	[x_seq, updates]=theano.scan(fn=forward_step,
									outputs_info=[x0],
									sequences=[betas],
									n_steps=nsteps)
	return x_seq, updates


def loss(x_0, n, t, params):
	muparams=params[:5]
	covparams=params[5:10]
	tpoints=T.cast(T.arange(nsteps),'float32')/T.cast(nsteps,'float32')
	betas=compute_betas(params[-1],tpoints)
	
	def step(nt, bt, xt):
		mean=xt*T.sqrt(1.0-bt)
		xnew=T.cast(mean+T.sqrt(bt)*nt,'float32')
		losst=T.cast(0.5*T.mean(T.sum((((mean-xnew)**2)/bt+T.log(np.pi*2.0*bt)),axis=1)),'float32')
		return xnew, losst
	
	[xhist, fwdlosshist],updates=theano.scan(fn=step,
								outputs_info=[x_0, None],
								sequences=[n, betas],
								n_steps=nsteps)
	
	
	forward_loss=-T.mean(fwdlosshist)+0.5*T.mean(T.sum((xhist[-1]**2+T.log(np.pi*2.0)),axis=1))
	
	#f_mu=compute_f_mu(xhist,t,muparams)
	#f_cov=compute_f_cov(xhist,t,covparams)
	#diffs=(f_mu[2:]-xhist[:-1])**2
	#gaussian_terms=T.sum(diffs*(1.0/f_cov[1:].dimshuffle(0,1,'x')),axis=2)
	#det_terms=T.sum(T.log(f_cov[1:].dimshuffle(0,1,'x')),axis=2)
	
	f_mu=compute_f_mu(xhist,t,muparams)+xhist*(T.sqrt(1.0-betas)).dimshuffle(0,'x','x')
	f_cov=compute_f_cov(xhist,t,covparams)*betas.dimshuffle(0,'x')
	xhist=T.concatenate([x_0.dimshuffle('x',0,1), xhist],axis=0)
	diffs=(f_mu-xhist[:-1])**2
	gaussian_terms=T.sum(diffs*(1.0/f_cov.dimshuffle(0,1,'x')),axis=2)
	det_terms=T.sum(T.log(f_cov.dimshuffle(0,1,'x')),axis=2)
	
	reverse_loss=T.mean(T.mean(gaussian_terms+det_terms))
	return reverse_loss+forward_loss


def get_loss_grad(params, x_0, n):
	
	t0=T.cast(T.arange(nsteps),'float32')/T.cast(nsteps,'float32')
	t=T.reshape(t0,(nsteps,1,1))
	t=T.extra_ops.repeat(t,x_0.shape[0],axis=1)
	objective=loss(x_0,n,t,params)
	gparams=T.grad(objective, params, consider_constant=[x_0,t,n])
	
	return objective, gparams


def reverse_step(beta, x, t, nsamps, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
	
	muparams=[p0, p1, p2, p3, p4]
	covparams=[p5, p6, p7, p8, p9]
	
	#f_mu=compute_f_mu(x,t,muparams)
	#f_cov=compute_f_cov(x,t,covparams)
	f_mu=compute_f_mu(x,t,muparams)+x*(T.sqrt(1.0-beta)).dimshuffle('x','x')
	f_cov=compute_f_cov(x,t,covparams)*beta.dimshuffle('x','x')
	
	samps=theano_rng.normal(size=(1,nsamps, nx))
	samps=samps*T.sqrt(f_cov).dimshuffle(0,1,'x')+f_mu
	return T.cast(samps,'float32'),T.cast(t-1.0/nsteps,'float32')


def get_samps(nsamps, params):
	
	t=1.0
	t=T.reshape(t,(1,1,1))
	t=T.extra_ops.repeat(t,nsamps,axis=1)
	t=T.cast(t,'float32')
	x0=theano_rng.normal(size=(nsamps, nx))
	x0=T.reshape(x0,(1,nsamps,nx))
	tpoints=T.cast(-T.arange(nsteps)+nsteps-1,'float32')/T.cast(nsteps,'float32')
	betas=compute_betas(params[-1],tpoints)
	[samphist, ts], updates=theano.scan(fn=reverse_step,
									sequences=[betas],
									outputs_info=[x0,t],
									non_sequences=[nsamps,params[0],params[1],params[2],params[3],params[4],params[5],
						params[6],params[7],params[8],params[9]],
									n_steps=nsteps)
	return samphist[:,0,:,:], ts[:,0], updates


def get_tgating():
	t0=T.cast(T.arange(nsteps),'float32')/T.cast(nsteps,'float32')
	t=T.reshape(t0,(nsteps,1,1))
	t=T.extra_ops.repeat(t,1,axis=1)
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,muWT)+mubT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
	tgating=T.reshape(tgating,(t.shape[0],t.shape[1],ntgates,1))
	return tgating
		

#compute_tgating=theano.function([],get_tgating()[:,0,:,0])

#tgates=compute_tgating()
#print tgates.shape
#pp.plot(tgates)
#pp.figure(2)
#pp.plot(np.sum(tgates,axis=1))
#pp.show()

### Making the swiss roll

data=np.random.rand(nsamps,2)*8.0+4.0
data=np.asarray([data[:,0]*np.cos(data[:,0]), data[:,0]*np.sin(data[:,0])])+np.random.randn(2,nsamps)*0.1
data=4.0*data.T

#nmix=2
#mixmeans=np.random.randn(nmix,nx)*0.0
#mixmeans[0,0]=12.0; mixmeans[1,0]=-12.0#; mixmeans[2,1]=12.0; mixmeans[3,1]=-12.0
#probs=np.random.rand(nmix)*0.0+1.0
#probs=probs/np.sum(probs)
#data=[]
#for i in range(nsamps):
	#midx=np.dot(np.arange(nmix),np.random.multinomial(1,probs))
	#nsamp=np.random.randn(nx)*(float(midx)+1.0)*1.0
	#data.append(mixmeans[int(midx)]+nsamp)

data=np.asarray(data, dtype='float32')
data=whiten(data)*1.0
#pp.figure(1)
#pp.suptitle('Data Samples')
#pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
#pp.scatter(data[:,0],data[:,1],c='b',alpha=0.2)
#pp.hexbin(data[:,0],data[:,1])
#pp.colorbar()

#pp.figure(2)
#pp.suptitle('Histogram: Data Density vs. Distance from Origin')
#pp.axes(xlim=(0.25,2.25),ylim=(0,5),xlabel='Distance from Origin',ylabel='Probability Density')
#pp.hist(np.sqrt(np.sum(data**2,axis=1)),50,normed=True)
#pp.show()

for i in range(nhid_mu):
	idx=np.random.randint(0,nsamps)
	mu_centers[:,i]=data[idx]

for i in range(nhid_cov):
	idx=np.random.randint(0,nsamps)
	cov_centers[:,i]=data[idx]

#pp.scatter(mu_centers[0,:],mu_centers[1,:]); pp.show()

if load_model==False:
	init_params=[mu_centers, mu_spreads, mu_biases, mu_M, mu_b,
				cov_centers, cov_spreads, cov_biases, cov_M, cov_b,
				betas]
else:
	f=open(load_fn,'rb')
	init_params=cp.load(f)
	f.close()

print init_params[-1]


xT=T.fmatrix()
betasT=T.fvector()
xseq, xseq_updates=compute_forward_trajectory(xT,betasT)
get_forward_traj=theano.function([xT,betasT],xseq,updates=xseq_updates,allow_input_downcast=True)


subfuncs=[]
for i in range(n_subfuncs):
	idxs=np.random.randint(nsamps-1,size=batchsize)
	noise=np.random.randn(nsteps,batchsize,2).astype(np.float32)
	subfuncs.append([np.asarray(data[idxs,:],dtype='float32'),noise])


# Compiling the loss and gradient function

[mu_centersT, mu_spreadsT, mu_biasesT, mu_MT, mu_bT,
	cov_centersT, cov_spreadsT, cov_biasesT, cov_MT, cov_bT]=[T.fmatrix(), T.fmatrix(), T.fvector(),
			T.fmatrix(), T.fmatrix(),T.fmatrix(), T.fmatrix(), T.fvector(),
			T.fmatrix(), T.fvector()]

paramsT=[mu_centersT, mu_spreadsT, mu_biasesT, mu_MT, mu_bT,
	cov_centersT, cov_spreadsT, cov_biasesT, cov_MT, cov_bT, betasT]

noiseT=T.ftensor3()
lossT, gradT=get_loss_grad(paramsT, xT, noiseT)

f_df_T=theano.function([mu_centersT, mu_spreadsT, mu_biasesT, mu_MT, mu_bT,
					cov_centersT, cov_spreadsT, cov_biasesT, cov_MT, cov_bT, betasT, xT, noiseT],
					[lossT,gradT[0],gradT[1],gradT[2],gradT[3],gradT[4],gradT[5],
					gradT[6],gradT[7],gradT[8],gradT[9],gradT[10]],
					allow_input_downcast=True,
					on_unused_input='warn')

def f_df(params, subfunc):
	[loss, grad0,grad1,grad2,grad3,grad4,grad5,
	grad6,grad7,grad8,grad9,grad10] = f_df_T(params[0],params[1],params[2],params[3],params[4],params[5],
						params[6],params[7],params[8],params[9],params[10],
						subfunc[0],subfunc[1])
	return loss, [grad0,grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10]


# Compiling the sampling function

samplesT, tT, sample_updates=get_samps(nsamps, paramsT)
sample_T=theano.function([mu_centersT, mu_spreadsT, mu_biasesT, mu_MT, mu_bT,
					cov_centersT, cov_spreadsT, cov_biasesT, cov_MT, cov_bT, betasT],
					samplesT,
					allow_input_downcast=True)

def sample(params):
	out = sample_T(params[0],params[1],params[2],params[3],params[4],params[5],
						params[6],params[7],params[8],params[9],params[10])
	return out

opt_params=init_params

testsubfunc=[data, np.random.randn(nsteps,nsamps,2).astype(np.float32)]

testloss, teststuff=f_df(init_params, testsubfunc)
print testloss
exit()

forward_data=get_forward_traj(data, init_params[-1])



if plot_reverse_process:
	samples=sample(opt_params)
	dsize=5
	alph=0.02
	#Reverse
	pp.figure(1, figsize=(2.5,2.5))
	#pp.suptitle('Reverse Process Samples at t=T')
	pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	pp.scatter(samples[0,:nsamps/10,0],samples[0,:nsamps/10,1],c='r',s=dsize,alpha=alph)
	pp.savefig('swiss_p_1.pdf')
	pp.close()
	pp.figure(2, figsize=(2.5,2.5))
	#pp.suptitle('Reverse Process Samples at t=T/2')
	pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	pp.scatter(samples[nsteps/2,:nsamps/10,0],samples[nsteps/2,:nsamps/10,1],c='r',s=dsize,alpha=alph)
	pp.savefig('swiss_p_half.pdf')
	pp.close()
	pp.figure(3, figsize=(2.5,2.5))
	#pp.suptitle('Reverse Process Samples at t=0')
	pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	pp.scatter(samples[-1,:nsamps/10,0],samples[-1,:nsamps/10,1],c='r',s=dsize,alpha=alph)
	pp.savefig('swiss_p_0.pdf')
	pp.close()
	#Forward
	pp.figure(4, figsize=(2.5,2.5))
	#pp.suptitle('Forward Process Samples at t=T')
	pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	pp.scatter(forward_data[0,:nsamps/10,0],forward_data[0,:nsamps/10,1],c='b',s=dsize,alpha=alph)
	pp.savefig('swiss_q_1.pdf')
	pp.close()
	pp.figure(5, figsize=(2.5,2.5))
	#pp.suptitle('Forward Process Samples at t=T/2')
	pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	pp.scatter(forward_data[nsteps/2,:nsamps/10,0],forward_data[nsteps/2,:nsamps/10,1],c='b',s=dsize,alpha=alph)
	pp.savefig('swiss_q_half.pdf')
	pp.close()
	pp.figure(6, figsize=(2.5,2.5))
	#pp.suptitle('Forward Process Samples at t=0')
	pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	pp.scatter(forward_data[-1,:nsamps/10,0],forward_data[-1,:nsamps/10,1],c='b',s=dsize,alpha=alph)
	pp.savefig('swiss_q_0.pdf')
	pp.close()
	#pp.figure(7, figsize=(2.5,2.5))
	#pp.suptitle('Histogram: Model Density vs. Distance from Origin')
	#pp.axes(xlim=(0.25,2.25),ylim=(0,5),xlabel='Distance from Origin',ylabel='Probability Density')
	#pp.hist(np.sqrt(np.sum(samples[-1]**2,axis=1)),50,normed=True,color='r')
	#pp.figure(8, figsize=(2.5,2.5))
	#pp.suptitle(r'Learned $\beta$ Schedule')
	#pp.axes(xlabel='t', ylabel=r'$\beta$')
	#pp.plot(np.arange(nsteps),(1.0/(1.0+np.exp(-opt_params[-1])))*beta_max)
	#pp.show()
	hst=2.25
	pp.figure(9, figsize=(2.5,2.5))
	pp.hexbin(samples[-1,:,0],samples[-1,:,1], extent=[-hst, hst, -hst, hst],gridsize=75)
	#pp.axis([-2,2,-2,2])
	pp.savefig('p_0_hist.pdf')
	pp.close()
	pp.figure(10, figsize=(2.5,2.5))
	pp.hexbin(data[:,0], data[:,1], extent=[-hst, hst, -hst, hst],gridsize=75)
	#pp.axis([-2,2,-2,2])
	pp.savefig('q_0_hist.pdf')
	pp.close()

exit()

if automate_training:
	optimizer = SFO(f_df, init_params, subfuncs)
	end_loss=99.0
	while end_loss>-2.50:
		linalgerror=False
		try:
			opt_params = optimizer.optimize(num_passes=2)
			end_loss = f_df(opt_params,fdata)[0]
		except np.linalg.linalg.LinAlgError:
			linalgerror=True
		
		if np.isnan(end_loss) or linalgerror:
			mu_centers=(np.random.randn(nx, nhid_mu)*1.0).astype(np.float32)
			mu_spreads=(np.zeros((nx, nhid_mu))-1.0).astype(np.float32)
			mu_biases=np.zeros(nhid_mu).astype(np.float32)
			mu_M=(np.random.randn(nhid_mu, ntgates*nx)*0.01).astype(np.float32)
			mu_b=np.zeros((ntgates, nx)).astype(np.float32)
			cov_centers=(np.random.randn(nx, nhid_cov)*1.0).astype(np.float32)
			cov_spreads=(np.zeros((nx, nhid_cov))-1.0).astype(np.float32)
			cov_biases=np.zeros(nhid_cov).astype(np.float32)
			cov_M=(np.random.randn(nhid_cov, ntgates*nx)*0.01).astype(np.float32)
			cov_b=np.zeros(ntgates).astype(np.float32)
			
			init_params=[mu_centers, mu_spreads, mu_biases, mu_M, mu_b,
						cov_centers, cov_spreads, cov_biases, cov_M, cov_b]
			
			optimizer = SFO(f_df, init_params, subfuncs)
			end_loss=99.0

else:
	# Creating the optimizer
	optimizer = SFO(f_df, init_params, subfuncs)
	old_params=init_params
	# Running the optimization
	init_loss = f_df(init_params,subfuncs[0])[0]
	print init_loss
	keyin=''
	while keyin!='y':
		opt_params = optimizer.optimize(num_passes=n_epochs)
		end_loss = f_df(opt_params,subfuncs[0])[0]
		samples=sample(opt_params)
		print samples.shape
		#pp.scatter(samples[-1,:,0],samples[-1,:,1])
		pp.hexbin(samples[-1,:,0],samples[-1,:,1])
		pp.colorbar()
		pp.figure(2)
		pp.plot(compute_betas_numpy(opt_params[-1]),color='b')
		pp.plot(compute_betas_numpy(old_params[-1]),color='r')
		pp.plot(beta_max,color='g')
		pp.show()
		print 'Current loss: ', end_loss
		print opt_params[-1]
		keyin=raw_input('End optimization? (y)')
		old_params=opt_params


if save_model_and_optimizer:
	f=open(save_fn,'wb')
	cp.dump(opt_params, f, 2)
	cp.dump(optimizer, f, 2)
	f.close()

x=np.arange(-3,3,0.1)
#locs=[]
#for i in x:
	#for j in x:
		#locs.append([i,j])
#locs=np.asarray(locs)
X,Y=np.meshgrid(x,x)
locs=np.asarray([X,Y]).T
[centers, spreads, biases, M, b]=opt_params[:5]
[covcenters, covspreads, covbiases, covM, covb]=opt_params[5:10]

## For plotting the RBF responses

diffs=locs.reshape((locs.shape[0], locs.shape[1], locs.shape[2], 1))-centers.reshape((1,1,centers.shape[0],centers.shape[1]))
scaled_diffs=(diffs**2)*np.exp(spreads).reshape((1,1,spreads.shape[0],spreads.shape[1]))
exp_terms=np.sum(scaled_diffs,axis=2)+biases.reshape((1,1,biases.shape[0]))
h=np.exp(exp_terms)
sumact=np.sum(h,axis=2)
#Normalization
hnorm=h/sumact.reshape((sumact.shape[0],sumact.shape[1],1))
fig=pp.figure(figsize=(8,8))
width=int(np.sqrt(hnorm.shape[2]))
for i in range(width):
	for j in range(width):
		ax=fig.add_subplot(width, width, i*width+j+1)
		#ax.matshow(hnorm[:,i*width+j].reshape((np.sqrt(hnorm.shape[0]),np.sqrt(hnorm.shape[0]))))
		ax.pcolor(Y, X, hnorm[:,:,i*width+j], vmin=0, vmax=1)

pp.show()
## For plotting the mu and covariance "fields"

locsT=T.ftensor3()
tT=T.fscalar()
optmuparamsT=[mu_centersT, mu_spreadsT, mu_biasesT, mu_MT, mu_bT]
vecsT=compute_mu_field(locsT, tT, optmuparamsT)
get_mu_field=theano.function([locsT, tT, mu_centersT, mu_spreadsT, mu_biasesT, mu_MT, mu_bT],
									vecsT,
									allow_input_downcast=True)

optcovparamsT=[cov_centersT, cov_spreadsT, cov_biasesT, cov_MT, cov_bT]
covsT=compute_cov_field(locsT, tT, optcovparamsT)
get_cov_field=theano.function([locsT, tT, cov_centersT, cov_spreadsT, cov_biasesT, cov_MT, cov_bT],
									covsT,
									allow_input_downcast=True)

t=0.0
vecfig=pp.figure(figsize=(8,8))
covfig=pp.figure(figsize=(8,8))
width=int(np.sqrt(ntgates))
vecfields=[]
covfields=[]
speeds=[]
covmags=[]
covmax=0
covmin=99999
speedmax=0
for i in range(ntgates):
		vecfield=get_mu_field(locs, t, centers, spreads, biases, M, b)
		covfield=get_cov_field(locs, t, covcenters, covspreads, covbiases, covM, covb)
		Umu=vecfield[:,:,0]
		Vmu=vecfield[:,:,1]
		vecfields.append(vecfield)
		covfields.append(covfield)
		speed = np.sqrt(Umu**2 + Vmu**2)
		speeds.append(speed)
		speedmax=np.maximum(speed.max(),speedmax)
		covmax=np.maximum(covfield.max(),covmax)
		covmin=np.minimum(covfield.min(),covmin)
		t=t+1.0/float(ntgates)

t=0.0
for i in range(width):
	for j in range(width):
		axmu=vecfig.add_subplot(width, width, i*width+j+1)
		axcov=covfig.add_subplot(width, width, i*width+j+1)
		vecfield=vecfields[i*width+j]
		covfield=covfields[i*width+j]
		speed=speeds[i*width+j]
		Umu=vecfield[:,:,0]
		Vmu=vecfield[:,:,1]
		lwmu = np.clip(30*speed/speedmax,0,5)
		axmu.streamplot(x, x, Umu.T, Vmu.T, density=0.6, color='k', linewidth=lwmu)
		axcov.pcolor(Y, X, covfield, vmin=covmin, vmax=covmax)
		t=t+1.0/float(ntgates)
	
pp.show()


if save_reverse_animation:
	samples=sample(opt_params)
	fig = pp.figure()
	ax = pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	paths = ax.scatter(samples[0,:,0],samples[0,:,1],c='r',alpha=.2)
	
	def init():
		paths.set_offsets(samples[0,:,:])
		return paths,
	
	# animation function.  This is called sequentially
	def animate(i):
		if i<nsteps:
			paths.set_offsets(samples[i,:,:])
		else:
			paths.set_offsets(samples[-1,:,:])
		return paths,
	
	anim = animation.FuncAnimation(fig, animate, init_func=init,
								   frames=nsteps+50, interval=100, blit=True)
	
	mywriter = animation.FFMpegWriter()
	anim.save('reverse_process.mp4', fps=2)


betas=opt_params[-1]
if save_forward_animation:
	fdata=get_forward_traj(data,betas)
	fig = pp.figure()
	ax = pp.axes(xlim=(-xlm, xlm), ylim=(-ylm, ylm))
	paths = ax.scatter(fdata[0,:,0],fdata[0,:,1],c='r',alpha=.2)

	def init():
		paths.set_offsets(fdata[0,:,:])
		return paths,

	# animation function.  This is called sequentially
	def animate(i):
		if i<nsteps:
			paths.set_offsets(fdata[i,:,:])
		else:
			paths.set_offsets(fdata[-1,:,:])
		return paths,

	anim = animation.FuncAnimation(fig, animate, init_func=init,
								   frames=nsteps+50, interval=100, blit=True)

	mywriter = animation.FFMpegWriter()
	anim.save('forward_process.mp4', fps=2)



