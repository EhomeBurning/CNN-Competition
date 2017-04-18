dataDir_train = fullfile('train','color');
dataDir_train_mask = fullfile('train','mask');
dataDir_train_normal = fullfile('train','normal');
% outputDir = fullfile('train','mynormal');
loadchallImages(dataDir_train);
disp('Images')
loadchallmask(dataDir_train_mask);
disp('mask')
loadchallgd(dataDir_train_normal);
disp('normal')