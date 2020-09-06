%img = imread('s1_1.bmp');
 I=imread('D:\MatlabFiles\FacelRecognition\HOG\36.png');
 img=rgb2gray(I);
% img = imread('caomei3.jpg');
[featureVector,hogVisualization] = extractHOGFeatures(img);
figure;
imshow(img);
hold on;
plot(hogVisualization);
