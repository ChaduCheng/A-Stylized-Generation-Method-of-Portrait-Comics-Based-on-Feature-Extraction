clear all;clc
close all

[filename,pathname] = uigetfile({'*.jpg;*.bmp;*.tif;*.png;*.gif','All Image Files'},'��ѡ��һ������ͼƬ');
if filename == 0%���û��ѡ��ֱ�ӷ��ؼ���
    return;
end
strfullname = strcat(pathname,filename);%ȡ��ͼ���ļ�ȫ��


I = imread(strfullname);%��ȡͼƬ

grayI = rgb2gray(I);
edgeI = edge(grayI,'canny',0.1);
figure
imshow(edgeI)
title('1��Ե���')

[h,w] = size(edgeI);

% ��ֵͼ����̬ѧ����
SE=strel('square',5);
% imopen�������ٸ�ʴ
P1 = imclose(edgeI,SE);

%�������������Ҳ�ױ� �Լ��ײ����ϰױ�
[y,x] = find(P1);
[minx,index] = min(x);%�׵�������
miny = y(index);
%ͼ�������ȫŪΪ1
for i=miny:h
    P1(i,minx) = 1;
end


[maxx,index] = max(x);%�׵�����Ҷ�
maxy = y(index);
%ͼ�����Ҳ�ȫŪΪ1
for i=maxy:h
    P1(i,maxx) = 1;
end

%ͼ��ײ�ŪΪ1
for i=minx:maxx
    P1(h,i) = 1;
end

figure
imshow(P1)
title('2��̬ѧ����')


[L,num] = bwlabel(P1,8);%�����ͨ����
STATS = regionprops(L,'BoundingBox');%ȡ��ÿ����ͨ����Ŀɰ�Χ����
areas = [];
for i = 1:num
    box = STATS(i).BoundingBox;
   area = box(3)*box(4);
   areas = [areas,area];
end
[maxvalue,maxindex] = max(areas);%�ҳ���������
P2 = zeros(h,w);
P2(find(L==maxindex))=1;%������ͨ���򴦵İ׵���1
P2 = logical(P2);
figure
imshow(P2)
title('3�����ͨ����')

% �׶����
P2 = imfill(P2,'holes');
figure
imshow(P2)
title('4��䴦��')
%��ʱP2���Ǵ�ԭͼ�ָ��������ӳ��

% ����P2�ָ������ͼ
segI = I;
[y,x] = find(P2 == 0);
num = length(y);
for i=1:num
    segI(y(i),x(i),:) = [0 0 0];
end
figure
imshow(segI)
title('5����ָ�')


%���������Ƿ�ɫ�ָ�
skinP = zeros(h,w);
ycbcr = rgb2ycbcr(segI);
figure
imshow(ycbcr)
title('ycbcr')
cr = double(ycbcr(:,:,3));%cr�ɷ�
cb =  double(ycbcr(:,:,2));%cb�ɷ�
for i=1:h
    for j=1:w
        if(133<=cr(i,j) && cr(i,j)<=173 && 77<=cb(i,j) && cb(i,j)<=127 )
            skinP(i,j) = 1;
        end
    end
end

figure
imshow(skinP)
title('6��ɫ�ָ�')

%������ ���������ҳ�����ɫ�������
% ��ֵͼ����̬ѧ����
SE=strel('square',5);
% imopen�ȸ�ʴ������
skinP = imopen(skinP,SE);
skinP = imfill(skinP,'holes');%�
[L,num] = bwlabel(skinP,8);%�����ͨ����
STATS = regionprops(L,'BoundingBox');%ȡ��ÿ����ͨ����Ŀɰ�Χ����
areas = [];
for i = 1:num
    box = STATS(i).BoundingBox;
   area = box(3)*box(4);
   areas = [areas,area];
end
[maxvalue,maxindex] = max(areas);%�ҳ���������
P3 = zeros(h,w);
P3(find(L==maxindex))=1;%������ͨ���򴦵İ׵���1
P3 = logical(P3);
SE=strel('square',5);
% imopen����
P3 = imdilate(P3,SE);
figure
imshow(P3)
title('7��ɫ����')


%��ʱP2������ָ�ͼ��P3�Ƿ�ɫ�����ָ�ͼ
P4 = zeros(h,w);
index1 = find(P2 == 1);%P2�а׵㲿������
Ptemp = P3(index1);%��P3���ҳ���Щ��
index2 = find(Ptemp == 0);%��Щ��P3��Ϊ0�ģ�����ʣ��ķ��ͷ���
P4(index1(index2)) = 1;

figure
imshow(P4)
title('8���ͷ���')






% rectangle('Position',maxbox,'EdgeColor','r');
