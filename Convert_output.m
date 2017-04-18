clc
close all
% prediction = uint16(prediction);
outputDir = fullfile('test','output');
parfor i = 1:size(prediction,1)
    A = prediction(i,:);
    x = A(1:128*128);
    y = A(128*128+1:2*128*128);
    z = A(2*128*128+1:end);
    x_t = reshape(x,[],128);
    y_t = reshape(y,[],128);
    z_t = reshape(z,[],128);
    com = zeros(128,128,3);
    com(:,:,1) = x_t;
    com(:,:,2) = y_t;
    com(:,:,3) = z_t;
    filename = fullfile(outputDir, sprintf('%d.png',i-1));
    imwrite(com,filename)
end