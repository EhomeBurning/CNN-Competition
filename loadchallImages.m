function loadchallImages(pathname)
d = dir(fullfile(pathname,'*.png'));
files = {d.name};
num = numel(files);
outputDir = fullfile('train','color');
for i=1:num
  image = im2double(rgb2gray(imread(fullfile(pathname, files{i}))));
  image = reshape(image,[],numel(image));
  filename = fullfile(outputDir, sprintf('%d.mat',i-1));
  save(filename,'image');
end