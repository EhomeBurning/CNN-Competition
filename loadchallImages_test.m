function loadchallImages_test(pathname)
d = dir(fullfile(pathname,'*.png'));
files = {d.name};
num = numel(files);
outputDir = fullfile('test','color_mat');
image = zeros(num,128*128);
filename = fullfile(outputDir, 'test_img.mat');
parfor i=1:num
  image_tmp = im2double(rgb2gray(imread(fullfile(pathname, files{i}))));
  image(i,:) = reshape(image_tmp,[],128*128);
end
save(filename,'image');