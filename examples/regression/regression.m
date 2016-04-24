function [net, info] = regression(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'regression') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = regression_init('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = genRegressionData(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

opts.train.errorFunction = 'regression';
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
%labels = imdb.images.labels(1,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
%labels = imdb.images.labels(1,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;


function imdb = genRegressionData(opts)

%Generate multinomial regression data 
x = randn(10,50000);
weights = randn(2,10);
bias = randn(1);
y = weights*x.^2 + bias;

trainTest = ones( 1,size(y,2));
idx = randsample( size(y,2), round( size(y,2) * 0.3));
trainTest(idx) = 3;

imdb.images.set = trainTest ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.images.data = single( permute( x, [4 1 3 2]));

imdb.images.data_mean = mean( imdb.images.data(:,:,:,trainTest == 1), 4);
imdb.images.data = bsxfun(@minus, imdb.images.data, single(imdb.images.data_mean));

%for Pdist loss :objective samples must have the size of the output tensor
imdb.images.labels = reshape( y, 1, 1, size(y,1), size(y,2) );

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

imdb.meta.classes = [];
