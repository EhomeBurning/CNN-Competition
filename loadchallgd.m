function loadchallgd(pathname)
d = dir(fullfile(pathname,'*.png'));
files = {d.name};
num = numel(files);
outputDir = fullfile('train','normal');
for i=1:num
  Im = im2double(imread(fullfile(pathname, files{i})));
  gd_truth_x = Im(:,:,1);
  gd_truth_y = Im(:,:,2);
  gd_truth_z = Im(:,:,3);
  gd_truth_x = reshape(gd_truth_x,[],numel(gd_truth_x));
  gd_truth_y = reshape(gd_truth_y,[],numel(gd_truth_y));
  gd_truth_z = reshape(gd_truth_z,[],numel(gd_truth_z));
  gd_truth = [gd_truth_x,gd_truth_y,gd_truth_z];
  filename = fullfile(outputDir, sprintf('%d.mat',i-1));
  save(filename,'gd_truth');
end

