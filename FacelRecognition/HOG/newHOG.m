%%matlabʵ��hog����
%�޸���http://www.cnblogs.com/tiandsp/archive/2013/05/24/3097503.html
%input: img
%output: final_descriptor

clear all; close all; clc;

%img=double(imread('lena.jpg'));
%img=imread('man.png');
img=imread('lemon.jpg');
%img=rgb2gray(img); %���������ͼת�Ҷ�ͼ���������ԸĽ���
img=imresize(img, [128 64]);
img=double(img);

[h, w, ~] = size(img);

%��������cell
cell_size=8;     %step*step��������Ϊһ��cell. cell_size=pixels_per_cell
orient=9;   %����ֱ��ͼ�����ķ�����
angle_range=180/orient; %ÿ����������ĽǶ���

h=round(h/cell_size)*cell_size;
w=round(w/cell_size)*cell_size;
img=img(1:h,1:w,:);

%img = sqrt(img); %٤��У����J=AI^r �˴�ȡA=1,r=0.5

% ���������Ե
fy=[-1 0 1]; %������ֱģ��
fx=fy';      %����ˮƽģ��

Gy=imfilter(img, fy, 'replicate'); %��ֱ�ݶ�
Gx=imfilter(img, fx, 'replicate'); %ˮƽ�ݶ�
Gmag=sqrt(Gx.^2+Gy.^2);            %�ݶȷ�ֵ

%Ϊÿ��cell������decriptor(�ݶȷ���ֱ��ͼ����һ��1*orient����������
cell_descriptors=zeros(orient, h/cell_size, w/cell_size);
idx_y=1;
for y=1:cell_size:h
    idx_x=1;
    for x=1:cell_size:w
        tmpx=Gx(y:y+cell_size-1, x:x+cell_size-1);
        tmpy=Gy(y:y+cell_size-1, x:x+cell_size-1);
        tmped=Gmag(y:y+cell_size-1,x:x+cell_size-1);
        tmped=tmped/sum(sum(tmped)); %�ֲ���Եǿ�ȹ�һ��
        cell_hist=zeros(1, orient); %��ǰcell_size*cell_size����ͳ�ƽǶ�ֱ��ͼ������cell
        for p=1:cell_size
            for q=1:cell_size
                ang=atan2(tmpy(p,q), tmpx(p,q));   %atan2���ص���[-pi,pi]֮��Ļ���ֵ
                ang=mod(ang*180/pi, 180);   %��ת�Ƕȣ��ٻ��鵽[0,180)֮�䡣��Ϊmod�Ĳ������ڲ�����������˻����179.
                ang=ang+0.0000001; %��ֹangΪ0
                
                bin_id = ceil(ang/angle_range);%�õ���bin_id \in [1,9]
                cell_hist(bin_id)=cell_hist(bin_id)+tmped(p,q); %ceil����ȡ����ʹ�ñ�Եǿ�ȼ�Ȩ���˴������ݶȷ������vote��ȨֵΪ�ݶȷ�ֵ
            end
        end
        cell_descriptors(:,idx_y,idx_x) = cell_hist;
        idx_x = idx_x + 1;
    end
    idx_y = idx_y + 1;
end    


%�����Ǽ���feature,block_size*block_size��cell�ϳ�һ��block
%����block_sizeȡ2
[~, h, w]=size(cell_descriptors);
block_size=2; %cells_per_block=2����ÿ��block_size=2*8=16����
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

%��block_descriptors����ƴ�ӣ��õ�final_descriptor
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