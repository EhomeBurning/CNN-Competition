function loadchallmask(pathname)
d = dir(fullfile(pathname,'*.png'));
files = {d.name};
% Im = cell(numel(files));
num = numel(files);
outputDir = fullfile('train','mask');
for i=1:num
  mask = im2double(imread(fullfile(pathname, files{i})));
  mask = reshape(mask,[],numel(mask));
  filename = fullfile(outputDir, sprintf('%d.mat',i-1));
  save(filename,'mask');
end