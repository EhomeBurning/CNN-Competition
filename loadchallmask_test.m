function [mask_rec] = loadchallmask_test(pathname)
d = dir(fullfile(pathname,'*.png'));
files = {d.name};
Im = cell(numel(files));
num = numel(files);
outputDir = fullfile('test','mask_mat');
filename = fullfile(outputDir, 'test_mask.mat');
parfor i=1:num
  Im{i} = im2double(imread(fullfile(pathname, files{i})));
end
[h,w] = size(Im{1});
mask_rec = zeros(num,3*h*w);
parfor i = 1:num
    mask = Im{i};
    mask = repmat(reshape(mask,[],numel(mask)),[1,3]);
    mask_rec(i,:) = mask;
end
save(filename,'mask_rec')