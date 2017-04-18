dataDir_test = fullfile('test','color');
dataDir_test_mask = fullfile('test','mask');
% outputDir = fullfile('train','mynormal');
loadchallImages_test(dataDir_test);
disp('Image loading complete')
[mask_rec] = loadchallmask_test(dataDir_test_mask);
disp('Mask loading complete')