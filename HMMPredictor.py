# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Code translated from DAn for training an HMM for chord recognition.
http://www.ee.columbia.edu/~dpwe/e4896/code/prac10/chords_code.zip
'''

# <codecell>

# Where does the repo live?
ROOT_DIR = '.'

# <codecell>

import numpy as np
import pprint
import scipy.linalg
import matplotlib.pyplot as plt

# <codecell>

import sklearn.linear_model
import sklearn.cross_validation
import sklearn.grid_search

# <codecell>

def model_logistic(data, labels):
    
    n_classes = np.unique(labels).size
    
    # If there's only one class, there's nothing to discriminate
    if n_classes < 2:
        return lambda x: 1
        
    splitter = sklearn.cross_validation.StratifiedKFold(labels, 3)
    
    parameters_sgd = {'alpha': [1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                      'class_weight': ['auto'], 'n_iter': [2], 'shuffle': [True]}
    
    h = sklearn.grid_search.GridSearchCV( sklearn.linear_model.SGDClassifier(loss='log'),
                                          parameters_sgd, n_jobs=4, cv=splitter)
    
    h.fit(data.T, labels)
    
    classifier = h.best_estimator_
    class_probability = np.mean(labels)
    
    def pdf(X, use_log=False):
        '''Returns p(y | x) / p(y)'''
        
        P = classifier.predict_proba(X.T) / class_probability
        
        if use_log:
            return np.log(P)
        else:
            return P
        
    return pdf

# <codecell>

def model_gaussian(data, labels, alpha=1e-4):
    '''Fits a gaussian model to the data.
    Returns a PDF function.

    Arguments
    ---------
    data  - d-by-N array (each column is an observation)
    labels - N array of booleans indicating which data points to include
    alpha - (optional) smoothing parameter for the covariance matrix

    Returns
    -------
    pdf(test_data, use_log=False) - function to evaluate the (log-)likelihood of test data
    '''
    
    # Grab the relevant data
    data  = data[:,labels]
    
    d, N  = data.shape
    
    mu    = data.mean(axis=1)
    sigma = alpha * np.eye(d) + np.cov(data)
    
    precision = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    logZ      = 0.5 * (np.log(np.abs(sigma_det)) + d * np.log( 2 * np.pi ) )
    
    def pdf(x, use_log=False):
        xhat  = x.T - mu
        mahal = np.sum( np.dot( xhat, precision )*xhat, 1 )
        
        ll    = -0.5 * mahal - logZ
        
        if use_log:
            return ll
        else:
            return np.exp( ll + 1e-16)
        
    return pdf

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
        
        z = np.sum( A )                                 #z = sum(A(:));
        #% Set any zeros to one before dividing
        #% This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
        s = z + (z==0)
        M = A/s
    else:
        #% Keith Battocchi - v. slow because of repmat
        #z=sum(A,dim);
        size      = np.array( A.shape )
        size[dim] = 1
        z         = np.sum( A, axis=dim ).reshape( size )
        s         = z + (z==0)
        L         = A.shape[dim]                        #L=size(A,dim);
        d         = len( A.shape )                      #d=length(size(A));
        v         = np.ones( d )                        #v=ones(d,1);
        v[dim]    = L                                   #v(dim)=L;
        c         = np.tile( s, v.T )                   #c=repmat(s,v');
        M         = A/c                                 #M=A./c;
    return M, z

# <codecell>

def recognize_chords(Chroma, Models, Transitions, Initial):
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
    
    
    nftrdims, nframes = Chroma.shape                                  #[nftrdims,nframes] = size(Chroma);
    nmodels           = len( Models )                                 #nmodels = length(Models);  % num models
    Liks              = np.zeros( (nmodels, nframes) )                #Liks = zeros(nmodels,nframes);
    
    #% evaluate each frame under all models
    
    for j in xrange( nmodels ):                                       #for j = 1:nmodels;
        Liks[j, :] = Models[j](Chroma)
        
    #% Evaluate viterbi path
    Labels = viterbi_path( Initial, Transitions, Liks )
    
    #% Make the labels be 0..24 
    return Labels, Liks

# <codecell>

def train_chord_models(Features, Labels, DIST=model_gaussian):
    '''
    % [Models, Transitions, Initial] = train_chord_models(TrainFileList)
    %     Train single-Gaussian models of chords by loading the Chroma
    %     features and the corresponding chord label data from each of
    %     the items named in the TrainFileList cell array.  Return
    %     Models as an array of Gaussian models (e.g. Models(i).mean
    %     and Models(i).sigma as the covariance), and a transition
    %     matrix in Transitions. 
    % 2010-04-07 Dan Ellis dpwe@ee.columbia.edu  after extractFeaturesAndTrain.m
    '''
    
    
    Labels_FLAT   = np.hstack( Labels )
    Features      = np.hstack(Features)
    # I'm not calculating a specific number of models - infer it from the labels.
    nmodels = np.unique( Labels_FLAT ).size
    
    #% global mean/covariance used for empty models
    global_model = DIST(Features, np.ones(Features.shape[1], dtype=bool))
        
    Models     = []
    
    #% Individual models for all chords
    
    for i in xrange( nmodels ):
        mask = Labels_FLAT == i
        if mask.any():
            Models.append(DIST(Features, mask))
        else:
            Models.append(global_model)
    
    #% Count the number of transitions in the label set
    
    Transitions = np.ones( (nmodels, nmodels), dtype=float )
    Initial      = np.ones( nmodels, dtype=float)
    
    for song_labels in Labels:
        Initial[song_labels[0]] += 1.0
        for s_from, s_to in zip(song_labels[:-1], song_labels[1:]):
            Transitions[s_from, s_to] += 1.0
    
    #Initial = Transitions.sum(axis=0)
    # Normalize the rows of the transition matrix
    Transitions = Transitions.dot(np.diag(Transitions.sum(axis=1)**-1))
    
    # Normalize the initial state distribution
    Initial = Initial / Initial.sum()
    
    
    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.bar(range(len(Initial)), Initial), plt.axis('tight'), plt.title('Initial-state distribution')
    plt.subplot(122)
    plt.imshow(Transitions, aspect='auto', interpolation='nearest', vmin=0, vmax=1.0), plt.colorbar()
    plt.title('Transition matrix')
    
    return Models, Transitions, Initial

# <codecell>

def viterbi_path(initial, transmat, obslik):
    '''
    % VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
    % path = viterbi(initial, transmat, obslik)
    %
    % Inputs:
    % initial(i) = Pr(Q(1) = i)
    % transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
    % obslik(i,t) = Pr(y(t) | Q(t)=i)
    %
    % Outputs:
    % path(t) = q(t), where q1 ... qT is the argmax of the above expression.
    '''

    #% delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
    #% psi(j,t) = the best predecessor state, given that we ended up in state j at t
    
    scaled = 1
    
    
    T = obslik.shape[1]                            #T     = size(obslik, 2);
                                                   #prior = prior(:);
    Q = initial.shape[0]                             #Q     = length(prior);
    
    
    delta = np.zeros( (Q, T) )                     #delta = zeros(Q,T);
    psi   = np.zeros( (Q, T), dtype=np.int )       #psi   = zeros(Q,T);
    path  = np.zeros( T )                          #path  = zeros(1,T);
    scale = np.ones( T )                           #scale = ones(1,T);
    
    t = 0
    
    delta[:, t] = initial*obslik[:, t]               #delta(:,t) = prior .* obslik(:,t);
    if scaled:
        delta[:, t], n = normalise( delta[:, t] )  #[delta(:,t), n] = normalise(delta(:,t));
        
        scale[t] = 1.0/n                           #scale(t) = 1/n;
    
    #% arbitrary value, since there is no predecessor to t=1
    
    psi[:, t] = 0                                  #psi(:,t) = 0; 
    
    
    for t in xrange( 1, T ):
        dt          = (delta[:, t-1]*transmat.T).T
        psi[:, t]   = np.argmax( dt, axis=0 )
        delta[:, t] = dt[psi[:, t], range(Q)]*obslik[:, t]
        
        if scaled:
            delta[:, t], n = normalise( delta[:, t] )
            scale[t]       = 1.0/n
    
    path[T - 1] = np.argmax( delta[:, T - 1] )
    p           = delta[path[T - 1], T - 1]
    
    for t in xrange( T - 2, -1, -1 ):
        path[t] = psi[path[t + 1], t + 1]
        
    return path

# <codecell>

def loadData( directory, featureType ):
    
    v_glob = glob.glob( os.path.join( ROOT_DIR, directory, '*-{}.npy'.format( featureType ) ) )
    v_glob.sort()
    
    
    vectors = [np.load(vf).T for vf in v_glob]
    
    l_glob = glob.glob( os.path.join( ROOT_DIR, directory, '*-labels-minmaj.npy' ) )
    l_glob.sort()
    
    labels = [np.load(lf) for lf in l_glob]
    
        
    #return np.hstack( vectors ), np.hstack( labels )
    return vectors, labels

def wrapToChroma( vectors ):
    vectors = np.dot( np.kron( np.eye( 12 ), np.ones( (1, 4) ) ), vectors )
    vMin = np.min( vectors, axis=1 )
    vMax = np.max( vectors, axis=1 )
    return ((vectors.T - vMin)/(vMax - vMin)).T

# <codecell>

def chromaluma_to_chroma(CL, use100=False):
    ''' Convert a matrix of chroma-luma vectors into chroma

    Arguments
    ---------
    CL : d-by-N matrix of CL vectors

    use100 : bool should we append 100hz-centered bass chroma?


    Returns
    -------
    Chroma : 12-by-N
    '''
    #% Adapting tuning - find which eighth-tone has the most peaks
    #[vv,xx] = max(Data);
    #semisoff = 0.9 - (mean(mod(xx+1,4)) - 1.5);

    xx = (np.argmax(CL, axis=0) + 2).astype(int)
    semis_off = 0.9 - np.mean(np.mod(xx, 4)) + 1.5
    
    d, N = CL.shape
    MM = np.zeros( (12, d) )
    for i in range(12):
        MM[i] = np.exp(-0.5 * ( ( 1 - (0.5 + 0.5 * np.cos((( -4 * i + semis_off) + np.arange(d))/48 * 2 * np.pi)))/ 0.001)**2)
                       
    Chroma = MM.dot(CL)
    
    if use100:
        wts = np.exp(-0.5 * ((np.arange(d) - 12) /  24)**2)
        Chroma = np.vstack([Chroma, MM.dot(np.diag(wts).dot(CL))])
        
    # Normalize chroma vectors
    Chroma = Chroma ** 0.25
    #Chroma = Chroma - Chroma.min(axis=0)
    z = np.max(Chroma, axis=0) + 1e-5
    Chroma = Chroma / z
    
    return Chroma

# <codecell>

def evaluate_model(Models, Transitions, Initial, testVectors, testLabels):
    predictedLabels = []
    for tv in testVectors:
        predictedLabels.extend(recognize_chords( tv, Models, Transitions, Initial )[0])
    
    testLabels = np.hstack(testLabels)
    
    print "######## Train={}, Test={}, Feature={}".format( train, test, feature )
    print sklearn.metrics.classification_report( testLabels, predictedLabels, target_names=labels )
    
    #plt.subplot(1,2,i)
    confusion = sklearn.metrics.confusion_matrix( testLabels, predictedLabels ).astype(float)
    confusion /= confusion.sum( axis=1, keepdims=True )
    confusion = confusion[:,labelSort][labelSort,:]
    plt.imshow( confusion, interpolation='nearest' , vmin=0.0, vmax=1.0)
    plt.yticks( range( 25 ), labels )
    plt.xticks( range( 25 ), labels, rotation=80 )
    plt.ylabel('True'), plt.xlabel('Predicted')
    plt.colorbar()
    plt.title( "Train={}\nTest={}\nFeature={}".format( train, test, feature ) )

# <codecell>

if __name__=="__main__":
    
    import os
    import glob
    import scipy.io
    import sklearn.metrics
    import sklearn.cross_validation
    import pickle
    
    MODEL = model_gaussian
    #MODEL = model_logistic
    with open( os.path.join(ROOT_DIR, 'Training_Scripts/minmaj_dict.pickle' )) as f:
        chord_classes, chord_keys = pickle.load( f )
        
    labels = [chord_classes[chord_keys[i]][0] for i in xrange(25)]
    labelSort = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(labels))]
    labels.sort()

    for feature in ['CL-magnitude']:#['wrapCL']:#['raw-compressed']:#, 'encoded-compressed']:
        for train in ['data/beatles/']:
            
            trainVectors, trainLabels = loadData( train, feature )
        
            if feature == 'wrapCL':
                trainVectors = map(wrapToChroma, trainVectors)
            elif feature == 'CL-magnitude':
                trainVectors = map(chromaluma_to_chroma, trainVectors)
            
            plt.figure( figsize=(16, 8) )
            
            i = 1
            for i_train, i_test in sklearn.cross_validation.ShuffleSplit(len(trainLabels), n_iterations=5, test_size=0.2):
                
                Models, Transitions, Initial = train_chord_models([trainVectors[i] for i in i_train], 
                                                                  [trainLabels[i] for i in i_train], DIST=MODEL )
                plt.figure()
                evaluate_model(Models, Transitions, Initial, [trainVectors[i] for i in i_test], 
                                                             [trainLabels[i] for i in i_test])
            
            
            plt.figure( figsize=(16, 8) )
            Models, Transitions, Initial = train_chord_models( trainVectors, trainLabels, DIST=MODEL )
            
            i = 1
            for test in ['data/beatles/']:#, 'data/uspop2002-npy']:
                
                testVectors, testLabels   = loadData( test, feature )
                
                if feature == 'wrapCL':
                    testVectors = map(wrapToChroma, testVectors)
                elif feature == 'CL-magnitude':
                    testVectors = map(chromaluma_to_chroma, testVectors)
                
                evaluate_model(Models, Transitions, Initial, testVectors, testLabels)
                i = i + 1
    plt.show()
pass

