# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Code translated from DAn for training an HMM for chord recognition.
http://www.ee.columbia.edu/~dpwe/e4896/code/prac10/chords_code.zip
'''

# <codecell>

import numpy as np
import pprint

# <codecell>

def gaussian_prob( x, m, C, use_log=0 ):
    '''
    % GAUSSIAN_PROB Evaluate a multivariate Gaussian density.
    % p = gaussian_prob(X, m, C, use_log)
    %
    % p(i) = N(X(:,i), m, C) if use_log = 0 (default)
    % p(i) = log N(X(:,i), m, C) if use_log = 1. This prevents underflow.
    %
    % If X has size dxN, then p has size Nx1, where N = number of examples
    '''

    # if length(m)==1 % scalar
    m = np.array( [m] ).flatten()
    if m.shape == ():
        # x = x(:)';
        x = x.reshape( (1, x.shape[0]) )
    #[d N] = size(x);
    d, N = x.shape
    #%assert(length(m)==d); % slow
    #m = m(:);
    m = m.reshape( (m.shape[0], 1) )
    #M = m*ones(1,N); % replicate the mean across columns
    M = np.dot( m, np.ones( (1, N) ) )
    #denom = (2*pi)^(d/2)*sqrt(abs(det(C)));
    denom = (2*np.pi)**(d/2.0)*np.sqrt( np.abs( np.linalg.det( C ) ) )
    #mahal = sum(((x-M)'*inv(C)).*(x-M)',2);   % Chris Bregler's trick
    mahal = np.sum( np.dot( (x - M).T, np.linalg.inv( C ) )*(x - M).T, 1 )
    #if any(mahal<0)
    if (mahal < 0).any():
        print 'mahal < 0 => C is not psd'
    if use_log:
        #p = -0.5*mahal - log(denom);
        p = -0.5*mahal - np.log( denom )
    else:
        #p = exp(-0.5*mahal) / (denom+eps);
        p = np.exp( -0.5*mahal )/(denom + 2.2204e-16)
    return p, mahal

# <codecell>

def normalise(A, dim=None):
    '''
    % NORMALISE Make the entries of a (multidimensional) array sum to 1
    % [M, c] = normalise(A)
    % c is the normalizing constant
    %
    % [M, c] = normalise(A, dim)
    % If dim is specified, we normalise the specified dimension only,
    % otherwise we normalise the whole array.
    '''
    if dim is None:
        #z = sum(A(:));
        z = np.sum( A )
        #% Set any zeros to one before dividing
        #% This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
        s = z + (z==0)
        M = A/s
    else:
        #% Keith Battocchi - v. slow because of repmat
        #z=sum(A,dim);
        size = np.array( A.shape )
        size[dim] = 1
        z = np.sum( A, axis=dim ).reshape( size )
        s = z + (z==0)
        #L=size(A,dim);
        L = A.shape[dim]
        #d=length(size(A));
        d = len( A.shape )
        #v=ones(d,1);
        v = np.ones( d )
        #v(dim)=L;
        v[dim] = L
        #c=repmat(s,v');
        c = np.tile( s, v.T )
        #M=A./c;
        M = A/c
    return M, z

# <codecell>

def recognize_chords(Chroma, Models, Transitions, Priors):
    '''
    % [Labels,Liks] = recognize_chords(Chroma, Models, Transitions)
    %     Take a 12xN array of chroma features Chroma, an array of 
    %     Gaussian models Models(i).mean and Models(i).sigma
    %     (covariance), and a transition matrix Transitions 
    %     and calculate the most likely (Viterbi) label sequence, 
    %     returned in Labels.  The likelihoods of each model for each
    %     frame are returned in Liks.
    % 2010-04-07 Dan Ellis dpwe@ee.columbia.edu after doChordID.m
    '''
    
    #[nftrdims,nframes] = size(Chroma);
    nftrdims, nframes = Chroma.shape
    #nmodels = length(Models);  % num models
    nmodels = len( Models )
    
    #Liks = zeros(nmodels,nframes);
    Liks = np.zeros( (nmodels, nframes) )
    #% evaluate each frame under all models
    #for j = 1:nmodels;
    for j in xrange( nmodels ):
        #Liks(j,:) = gaussian_prob(Chroma, Models(j).mean, Models(j).sigma); 
        Liks[j, :], _ = gaussian_prob( Chroma, Models[j]['mean'], Models[j]['sigma'] )
    
    #% Evaluate viterbi path
    Labels = viterbi_path( Priors, Transitions, Liks )
    #% Make the labels be 0..24 
    return Labels, Liks

# <codecell>

def train_chord_models(Features, Labels):
    '''
    % [Models, Transitions, Priors] = train_chord_models(TrainFileList)
    %     Train single-Gaussian models of chords by loading the Chroma
    %     features and the corresponding chord label data from each of
    %     the items named in the TrainFileList cell array.  Return
    %     Models as an array of Gaussian models (e.g. Models(i).mean
    %     and Models(i).sigma as the covariance), and a transition
    %     matrix in Transitions. 
    % 2010-04-07 Dan Ellis dpwe@ee.columbia.edu  after extractFeaturesAndTrain.m
    '''
    
    # I'm not calculating a specific number of models - infer it from the labels.
    nmodels = np.unique( Labels ).shape[0]
    
    #% global mean/covariance used for empty models
    #globalmean = mean(Features')';
    globalmean = np.mean( Features, axis = 1 )
    #globalcov = cov(Features')';
    globalcov = np.cov( Features )
    
    Models = {}
    
    #% Individual models for all chords
    #for i = 1:nmodels
    for i in xrange( nmodels ):
        Models[i] = {}
        #examples = find(Labels == i-1);  % labels are 0..24
        examples = np.flatnonzero( Labels == i )
        #if length(examples) > 0
        if examples.shape[0] > 0:
            #% mean and cov expect data in columns, not rows - transpose twice
            #Models(i).mean = mean(Features(:,examples)')';
            Models[i]['mean'] = np.mean( Features[:, examples], axis = 1 )
            #Models(i).sigma = cov(Features(:,examples)')';
            Models[i]['sigma'] = np.cov( Features[:, examples] )
        else:
            #Models(i).mean = globalmean;
            Models[i]['mean'] = globalmean
            #Models(i).sigma = globalcov;
            Models[i]['sigma'] = globalcov;
    
    #% Count the number of transitions in the label set
    #% (transitions between tracks get factored in ... oh well)
    #% Each element of gtt is a 4 digit number indicating one transition 
    #% e.g. 2400 for 24 -> 0
    #gtt = 100*Labels(1:end-1)+Labels(2:end);
    gtt = 100*Labels[:-1] + Labels[1:]
    #% arrange these into the transition matrix by counting each type
    #Transitions = zeros(nmodels,nmodels);
    Transitions = np.zeros( (nmodels, nmodels) )
    for i in xrange( nmodels ):
        for j in xrange( nmodels ):
            #nn = 100*(i-1)+(j-1); 
            nn = 100*i + j
            #% Add one to all counts, so no transitions have zero probability
            #Transitions(i,j) = 1+sum(gtt==nn);
            Transitions[i, j] = 1 + np.sum( gtt == nn )
    
    #% priors of each chord
    #Priors = sum(Transitions,2);
    Priors = np.sum( Transitions, axis=1 )
    #% normalize each row
    #Transitions = Transitions./repmat(Priors,1,nmodels);
    Transitions = Transitions/np.tile( Priors[np.newaxis].T, (1, nmodels) )
    #% normalize priors too
    #Priors = Priors/sum(Priors);
    Priors /= Priors.sum()
    return Models, Transitions, Priors

# <codecell>

def viterbi_path(prior, transmat, obslik):
    '''
    % VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
    % path = viterbi(prior, transmat, obslik)
    %
    % Inputs:
    % prior(i) = Pr(Q(1) = i)
    % transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
    % obslik(i,t) = Pr(y(t) | Q(t)=i)
    %
    % Outputs:
    % path(t) = q(t), where q1 ... qT is the argmax of the above expression.
    '''


    #% delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
    #% psi(j,t) = the best predecessor state, given that we ended up in state j at t
    
    scaled = 1
    
    #T = size(obslik, 2);
    T = obslik.shape[1]
    #prior = prior(:);
    #Q = length(prior);
    Q = prior.shape[0]
    
    #delta = zeros(Q,T);
    delta = np.zeros( (Q, T) )
    #psi = zeros(Q,T);
    psi = np.zeros( (Q, T) )
    #path = zeros(1,T);
    path = np.zeros( T )
    #scale = ones(1,T);
    scale = np.ones( T )
    
    t = 1
    #delta(:,t) = prior .* obslik(:,t);
    delta[:, t] = prior*obslik[:, t]
    if scaled:
        #[delta(:,t), n] = normalise(delta(:,t));
        delta[:, t], n = normalise( delta[:, t] )
        #scale(t) = 1/n;
        scale[t] = 1.0/n
    #psi(:,t) = 0; % arbitrary value, since there is no predecessor to t=1
    psi[:, t] = 0
    #for t=2:T
    for t in xrange( 1, T ):
        #for j=1:Q
        for j in xrange( Q ):
            #[delta(j,t), psi(j,t)] = max(delta(:,t-1) .* transmat(:,j));
            delta[j, t] = np.max( delta[:, t-1]*transmat[:, j] )
            psi[j, t] = np.argmax( delta[:, t-1]*transmat[:, j] )
            #delta(j,t) = delta(j,t) * obslik(j,t);
            delta[j, t] *= obslik[j, t]
        if scaled:
            #[delta(:,t), n] = normalise(delta(:,t));
            delta[:, t], n = normalise( delta[:, t] )
            #scale(t) = 1/n;
            scale[t] = 1.0/n
    #[p, path(T)] = max(delta(:,T));
    path[T - 1] = np.argmax( delta[:, T - 1] )
    p = delta[path[T - 1], T - 1]
    #for t=T-1:-1:1
    for t in xrange( T - 2, -1, -1 ):
        #path(t) = psi(path(t+1),t+1);
        path[t] = psi[path[t + 1], t + 1]
    return path

# <codecell>

if __name__=="__main__":
    import os
    import glob
    import scipy.io
    
    def loadData( directory ):
        vectors = []
        labels = []
        for vectorFile in glob.glob( os.path.join( directory, '*wrapCL.npy' ) ):
            vectors.append( np.load( vectorFile ).T )
        for labelFile in glob.glob( os.path.join( directory, '*labels-minmaj.npy' ) ):
            labels.append( np.load( labelFile ) )
        return np.hstack( vectors ), np.hstack( labels )
    
    trainVectors, trainLabels = loadData( 'beatles' )
    
    scipy.io.savemat( 'train.mat', {'trainVectors':trainVectors, 'trainLabels':trainLabels} )
    
    # Pretty sure this step is right
    Models, Transitions, Priors = train_chord_models( trainVectors, trainLabels )
    #testVectors, testLabels = loadData( 'uspop2002-npy' )
    #Labels, Liks = recognize_chords( trainVectors, Models, Transitions, Priors )
    #print np.sum( Labels == trainLabels )/(1.0*Labels.shape[0])

