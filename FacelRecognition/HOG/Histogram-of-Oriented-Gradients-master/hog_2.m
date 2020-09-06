% im = imread('F:\matlabfile\HOG\Lena.jpg');
im=imread('cslbp2.jpg');
% im=imread('s1_1.bmp');
imshow(im);
figure();
D(:,:,1) = adapthisteq(im(:,:,1));
D(:,:,2) = adapthisteq(im(:,:,2));
D(:,:,3) = adapthisteq(im(:,:,3));
imshow(D);

figure();
hog = hog_feature_vector(D);