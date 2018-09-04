clear
clc
close all

%% 读取图片部分

I=imread('pic\9.jpg');
[M,N,C]=size(I);
figure
imshow(I);
title('原图')

% 获取灰度图
im = im2double(I);
if C>1 
    
    % ycbcr空间，y平面
    Iruv = rgb2ycbcr(im);
    I_gray=Iruv(:,:,1);
else
    I_gray=im;
end

%% 计算S图像部分

% the length of convolution line 
ks = 13; 
% the number of directions 
dirNum = 8;


% 得到S图像
S= GenStroke(I_gray,ks,dirNum);
figure
imshow(S)
title('S图像')

%% 直方图匹配部分


J=I_gray;

% 根据论文数据获取直方图
a=1:255;
p1 = 1/9*exp(-(255-a)/9);
p3=1/sqrt(2*pi*11) * exp(-(a-80).*(a-80) / (2.0*11*11));
p2= 1:255;
p2(:)=1/(225-105);
p2(1:105)=0;
p2(225:255)=0;
p = 0.52 * p1 + 0.37 * p2 + 0.11 * p3;
figure
plot(a,p)
title('直方图')

% 直方图匹配
J=histeq(J,p);
figure
imshow(J)
title('直方图匹配')

%% 纹理图像

% 读取纹理图像
P=imread('texture pic\pencil0.jpg');
[M2,N2,C2]=size(P);
if C2>1
    P=rgb2gray(P);
end
% 调整大小
P=imresize(P,[M N]);
P = im2double(P); 

figure
imshow(P)
title('纹理图像')

%% 纹理渲染部分

T = GenPencil(I_gray, P, J);
T=im2double(T);
figure
imshow(T)
title('纹理渲染')

%% 结果部分

res=T.*S;
figure
imshow(res);
title('结果图')

% 显示彩色部分
if C>1
    
    I_color_res(:,:,1)=res;
    I_color_res(:,:,2:3)=Iruv(:,:,2:3);
    % 转换回rgb空间
    I_color_res = ycbcr2rgb(I_color_res);
    % 结果显示
    figure
    imshow(I_color_res);
    title('结果颜色图')
end

