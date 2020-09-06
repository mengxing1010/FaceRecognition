%function [hog] = HOG(im)
image = imread('D:\MatlabFiles\FacelRecognition\HOG\CSLBPlemons.png');
% image=imread('F:\matlabfile\HOG\XCS-LBP-master\1.bmp');
gray_img = rgb2gray(image);
% gray_img=image;

%1: Convert the gray-scale image to double format.
img = im2double(gray_img);
whos img

%2: Get differential images using GetDifferentialFilter and FilterImage
[filter_x, filter_y] = GetDifferentialFilter();
im_dx = FilterImage(img, filter_x);
im_dy = FilterImage(img, filter_y);

%3: Compute the gradients using GetGradient
[grad_mag, grad_angle] = GetGradient(im_dx, im_dy);
% subplot(2,2,1);
figure;
imshow(im_dx);title('水平方向梯度');

% subplot(2,2,2);
figure;
imshow(im_dy);title('垂直方向梯度');
% subplot(2,2,3);
figure;
imshow(grad_mag);title('HoG特征图');
% subplot(2,2,4);
imshow(grad_angle);title('这个')


%drawing vectors
% x = grad_mag.*cos(grad_angle);
% y = grad_mag.*sin(grad_angle);
% dx = gradient(x);
% dy = gradient(y);


%4: Build the histogram of oriented gradients for all cells using BuildHistogram
%cell_size = 8;
%ori_histo = BuildHistogram(grad_mag, grad_angle, cell_size);


%5: Build the descriptor of all blocks with normalization using GetBlockDescriptor
%6: Return a long vector (hog) by concatenating all block descriptors.
