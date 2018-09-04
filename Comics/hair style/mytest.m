clear all;clc
close all

[filename,pathname] = uigetfile({'*.jpg;*.bmp;*.tif;*.png;*.gif','All Image Files'},'请选择一张人脸图片');
if filename == 0%如果没有选择，直接返回即可
    return;
end
strfullname = strcat(pathname,filename);%取得图像文件全名


I = imread(strfullname);%读取图片

grayI = rgb2gray(I);
edgeI = edge(grayI,'canny',0.1);
figure
imshow(edgeI)
title('1边缘检测')

[h,w] = size(edgeI);

% 二值图像形态学处理
SE=strel('square',5);
% imopen先膨胀再腐蚀
P1 = imclose(edgeI,SE);

%补充最左侧和最右侧白边 以及底部补上白边
[y,x] = find(P1);
[minx,index] = min(x);%白点的最左端
miny = y(index);
%图像最左侧全弄为1
for i=miny:h
    P1(i,minx) = 1;
end


[maxx,index] = max(x);%白点的最右端
maxy = y(index);
%图像最右侧全弄为1
for i=maxy:h
    P1(i,maxx) = 1;
end

%图像底部弄为1
for i=minx:maxx
    P1(h,i) = 1;
end

figure
imshow(P1)
title('2形态学处理')


[L,num] = bwlabel(P1,8);%标记连通区域
STATS = regionprops(L,'BoundingBox');%取得每个连通区域的可包围矩形
areas = [];
for i = 1:num
    box = STATS(i).BoundingBox;
   area = box(3)*box(4);
   areas = [areas,area];
end
[maxvalue,maxindex] = max(areas);%找出最大的区域
P2 = zeros(h,w);
P2(find(L==maxindex))=1;%最大的连通区域处的白点置1
P2 = logical(P2);
figure
imshow(P2)
title('3最大连通区域')

% 孔洞填充
P2 = imfill(P2,'holes');
figure
imshow(P2)
title('4填充处理')
%此时P2就是从原图分割出的人体映射

% 依靠P2分割出人物图
segI = I;
[y,x] = find(P2 == 0);
num = length(y);
for i=1:num
    segI(y(i),x(i),:) = [0 0 0];
end
figure
imshow(segI)
title('5人体分割')


%接下来就是肤色分割
skinP = zeros(h,w);
ycbcr = rgb2ycbcr(segI);
figure
imshow(ycbcr)
title('ycbcr')
cr = double(ycbcr(:,:,3));%cr成分
cb =  double(ycbcr(:,:,2));%cb成分
for i=1:h
    for j=1:w
        if(133<=cr(i,j) && cr(i,j)<=173 && 77<=cb(i,j) && cb(i,j)<=127 )
            skinP(i,j) = 1;
        end
    end
end

figure
imshow(skinP)
title('6肤色分割')

%接下来 处理，并且找出最大肤色区域出来
% 二值图像形态学处理
SE=strel('square',5);
% imopen先腐蚀再膨胀
skinP = imopen(skinP,SE);
skinP = imfill(skinP,'holes');%填洞
[L,num] = bwlabel(skinP,8);%标记连通区域
STATS = regionprops(L,'BoundingBox');%取得每个连通区域的可包围矩形
areas = [];
for i = 1:num
    box = STATS(i).BoundingBox;
   area = box(3)*box(4);
   areas = [areas,area];
end
[maxvalue,maxindex] = max(areas);%找出最大的区域
P3 = zeros(h,w);
P3(find(L==maxindex))=1;%最大的连通区域处的白点置1
P3 = logical(P3);
SE=strel('square',5);
% imopen膨胀
P3 = imdilate(P3,SE);
figure
imshow(P3)
title('7肤色处理')


%此时P2是人体分割图，P3是肤色人脸分割图
P4 = zeros(h,w);
index1 = find(P2 == 1);%P2中白点部分索引
Ptemp = P3(index1);%在P3中找出这些点
index2 = find(Ptemp == 0);%这些在P3中为0的，就是剩余的发型服饰
P4(index1(index2)) = 1;

figure
imshow(P4)
title('8发型服饰')






% rectangle('Position',maxbox,'EdgeColor','r');
