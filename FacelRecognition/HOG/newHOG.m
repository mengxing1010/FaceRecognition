%%matlab实现hog特征
%修改自http://www.cnblogs.com/tiandsp/archive/2013/05/24/3097503.html
%input: img
%output: final_descriptor

clear all; close all; clc;

%img=double(imread('lena.jpg'));
%img=imread('man.png');
img=imread('lemon.jpg');
%img=rgb2gray(img); %简单起见，彩图转灰度图。后续可以改进。
img=imresize(img, [128 64]);
img=double(img);

[h, w, ~] = size(img);

%下面是求cell
cell_size=8;     %step*step个像素作为一个cell. cell_size=pixels_per_cell
orient=9;   %方向直方图包含的方向数
angle_range=180/orient; %每个方向包含的角度数

h=round(h/cell_size)*cell_size;
w=round(w/cell_size)*cell_size;
img=img(1:h,1:w,:);

%img = sqrt(img); %伽马校正。J=AI^r 此处取A=1,r=0.5

% 下面是求边缘
fy=[-1 0 1]; %定义竖直模版
fx=fy';      %定义水平模版

Gy=imfilter(img, fy, 'replicate'); %竖直梯度
Gx=imfilter(img, fx, 'replicate'); %水平梯度
Gmag=sqrt(Gx.^2+Gy.^2);            %梯度幅值

%为每个cell计算其decriptor(梯度方向直方图，即一个1*orient规格的向量）
cell_descriptors=zeros(orient, h/cell_size, w/cell_size);
idx_y=1;
for y=1:cell_size:h
    idx_x=1;
    for x=1:cell_size:w
        tmpx=Gx(y:y+cell_size-1, x:x+cell_size-1);
        tmpy=Gy(y:y+cell_size-1, x:x+cell_size-1);
        tmped=Gmag(y:y+cell_size-1,x:x+cell_size-1);
        tmped=tmped/sum(sum(tmped)); %局部边缘强度归一化
        cell_hist=zeros(1, orient); %当前cell_size*cell_size像素统计角度直方图，就是cell
        for p=1:cell_size
            for q=1:cell_size
                ang=atan2(tmpy(p,q), tmpx(p,q));   %atan2返回的是[-pi,pi]之间的弧度值
                ang=mod(ang*180/pi, 180);   %先转角度，再划归到[0,180)之间。因为mod的参数现在不是整数，因此会大于179.
                ang=ang+0.0000001; %防止ang为0
                
                bin_id = ceil(ang/angle_range);%得到的bin_id \in [1,9]
                cell_hist(bin_id)=cell_hist(bin_id)+tmped(p,q); %ceil向上取整，使用边缘强度加权。此处根据梯度方向进行vote，权值为梯度幅值
            end
        end
        cell_descriptors(:,idx_y,idx_x) = cell_hist;
        idx_x = idx_x + 1;
    end
    idx_y = idx_y + 1;
end    


%下面是计算feature,block_size*block_size个cell合成一个block
%比如block_size取2
[~, h, w]=size(cell_descriptors);
block_size=2; %cells_per_block=2，即每个block_size=2*8=16像素
stride=1;
h_max=floor((h-block_size)/stride)+1;
w_max=floor((w-block_size)/stride)+1;
block_descriptors=zeros(block_size*block_size*orient, h_max, w_max);
for i=1:h_max
    for j=1:w_max
        blk_mat=cell_descriptors(:,i:i+block_size-1, j:j+block_size-1);
        normed_blk_mat=zz_normalize(blk_mat);
        reshaped_blk_mat=reshape(normed_blk_mat, [1 block_size*block_size*orient]);
        block_descriptors(:,i,j)=reshaped_blk_mat;
    end
end

%将block_descriptors进行拼接，得到final_descriptor
[d1,d2,d3]=size(block_descriptors);
dimensions=d1*d2*d3;
final_descriptor=zeros(1, dimensions);
k=1;
for i=1:d2
    for j=1:d3
        final_descriptor(k:k+d1-1)=block_descriptors(:,i,j);
        k=k+d1;
    end
end