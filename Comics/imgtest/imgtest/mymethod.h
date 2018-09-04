
#include <opencv2/opencv.hpp>
#include "asmfitting.h"
#include "vjfacedetect.h"

#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_objdetect249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "asmlibraryD.lib")

using namespace std;
using namespace cv;


enum PROCTYPE//���������
{
	PROC_ORI,//ԭʼ
	PROC_GRAY,//�ҶȻ�
	PROC_BW,//��ֵ��
	PROC_EDGE//��Ե
};

int imageResize(IplImage** pi,CvSize sz)
{
	IplImage *src = *pi;
	if((src->width == sz.width) && (src->height == sz.height)) //����ߴ籾������ȣ��������κα䶯
	{
		return 0;
	}
	IplImage *dst = cvCreateImage(sz,src->depth,src->nChannels);//����Ŀ���С
	cvResize(*pi,dst);//�����ߴ�

	cvReleaseImage(pi);     
	*pi = dst;
	return (1); 
}

int  imageReplace(IplImage* pi,IplImage** ppo)  //  IplImage�滻
{
	if (*ppo) 
		cvReleaseImage(ppo);                //  �ͷ�ԭ��λͼ
	(*ppo) = pi;                            //  λͼ����
	return(1);
}

int  imageClone(IplImage* pi,IplImage** ppo)  //  ���� IplImage λͼ
{
	if (*ppo) {
		cvReleaseImage(ppo);                //  �ͷ�ԭ��λͼ
	}
	(*ppo) = cvCloneImage(pi);              //  ������λͼ
	return(1);
}


void rgb2gray(IplImage* pi,IplImage** ppo)
{
	if (pi->nChannels == 1)//����Ѿ��ǻҶ�ͼ��
	{
		imageClone(pi,ppo);//ֱ�Ӹ���
	}
	else
	{
		if (*ppo) 
			cvReleaseImage(ppo);                //  �ͷ�ԭ��λͼ
		*ppo = cvCreateImage(cvGetSize(pi), 8, 1);
		cvCvtColor(pi,*ppo,CV_BGR2GRAY);//rgbתΪ�Ҷ�ͼ
	}
}

//OTSU�Զ���ֵ��
int ThresholdOtsu(IplImage* src, IplImage* dst, int type)
{
	int height=src->height;
	int width=src->width; 
	int i;
	//histogram
	float histogram[256]={0};
	for(i=0;i<height;i++)
	{
		unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;
		for(int j=0;j<width;j++)
		{
			histogram[*p++]++;
		}
	}

	//normalize histogram
	int size=height*width;
	for(i=0;i<256;i++)
	{
		histogram[i]=histogram[i]/size;
	}

	//average pixel value
	float avgValue=0;
	for(i=0;i<256;i++) 
	{
		avgValue+=i*histogram[i];
	}

	int threshold; 
	float maxVariance=0;
	float w=0,u=0;
	for(i=0;i<256;i++)
	{
		w+=histogram[i];
		u+=i*histogram[i];

		float t=avgValue*w-u;
		float variance=t*t/(w*(1-w));
		if(variance>maxVariance)
		{
			maxVariance=variance;
			threshold=i;
		}
	}
	//	threshold = threshold-10;
	cvThreshold(src,dst,threshold,255, type);
	return  threshold;
} 

//�������������ͼ��ʹ�����������������Ѱ�����ڵ�ģ�Ҳ�����ҽ����׵��
void cropimage(IplImage* inimage,IplImage** outimage,int type)
{
	IplImage* gray = NULL;//�Ҷ�ͼ��
	rgb2gray(inimage,&gray);//תΪ�Ҷ�ͼ��

	//Ȼ���ֵ��
	IplImage *bwimage = cvCreateImage(cvGetSize(gray), 8, 1);//����ռ�
	ThresholdOtsu(gray,bwimage,type);	//�Զ���ֵ��ֵ��

	int left,right,top,bottom;
	//��߽�
	for (int i=0;i<bwimage->width;i++)//����ÿһ��
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->height;j++)//����ÿһ��
		{
			double value = cvGetReal2D(bwimage,j,i);//��ȡ����ֵ
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//�ҵ��ڵ�����
		{
			left = i;//��߽�
			break;
		}

	}

	//�ұ߽�
	for (int i=bwimage->width-1; i>=0; i--)//����ÿһ��
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->height;j++)//����ÿһ��
		{
			double value = cvGetReal2D(bwimage,j,i);//��ȡ����ֵ
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//�ҵ��ڵ�����
		{
			right = i;//�ұ߽�
			break;
		}
	}

	//�ϱ߽�
	for (int i=0;i<bwimage->height;i++)//����ÿһ��
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->width;j++)//����ÿһ��
		{
			double value = cvGetReal2D(bwimage,i,j);//��ȡ����ֵ
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//�ҵ��ڵ�����
		{
			top = i;//��߽�
			break;
		}

	}

	//�±߽�
	for (int i=bwimage->height-1; i>=0; i--)//����ÿһ��
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->width;j++)//����ÿһ��
		{
			double value = cvGetReal2D(bwimage,i,j);//��ȡ����ֵ
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//�ҵ��ڵ�����
		{
			bottom = i;//�±߽�
			break;
		}
	}

	CvRect r;//��֯��������
	r.x = left;
	r.y = top;
	r.height = bottom-top+1;
	r.width = right-left+1;//
	cvSetImageROI(inimage,r);//��ͼ��ѡ������,��ȡ����
	*outimage = cvCreateImage( cvGetSize(inimage),8, inimage->nChannels);
	cvCopy(inimage,*outimage);
	cvResetImageROI(inimage);

	cvReleaseImage(&gray);
	cvReleaseImage(&bwimage);
}

//�������ʵ������
Mat sketch(Mat src)
{
	Mat dst = src.clone();
	int width=src.cols;  
	int heigh=src.rows;  

	Mat gray1;
	//��ɫ  
	addWeighted(src,-1,NULL,0,255,gray1);  
	//��˹ģ��,��˹�˵�Size������Ч���й�  
	GaussianBlur(gray1,gray1,Size(11,11),0);  
	//�ںϣ���ɫ����  
	for (int y=0; y<heigh; y++)  
	{  

		uchar* P0  = src.ptr<uchar>(y);  
		uchar* P1  = gray1.ptr<uchar>(y);  
		uchar* P  = dst.ptr<uchar>(y);  
		for (int x=0; x<width; x++)  
		{  
			int tmp0=P0[x];  
			int tmp1=P1[x];  
			P[x] =(uchar) min((tmp0+(tmp0*tmp1)/(256-tmp1)),255);  
		}  

	}  

	medianBlur(dst, dst, 3);
	return dst;
}

Mat sketch2(Mat src)
{
	Mat dst = src.clone();
	
	medianBlur(src, src, 7);
	/*cvLaplace(m_procimgcopy,m_procimgcopy);*/
	Laplacian(src, dst, CV_8U, 5);

	return dst;
}

//Ԥ����,��Ҫ��תΪ�Ҷȣ���ֵ������С��һ�ȵ�
void preproc(IplImage* inimage,IplImage** outimage,CvSize siz,PROCTYPE type)
{
	
	if(type == PROC_ORI)
	{
		cropimage(inimage,outimage,CV_THRESH_BINARY);//�ҵ������ĺڱߣ����г���
		imageResize(outimage,siz);//������ͳһ��С
	}
	if(type == PROC_GRAY)
	{
		IplImage* gray = NULL;//�Ҷ�ͼ��
		rgb2gray(inimage,&gray);//תΪ�Ҷ�ͼ��
		imageResize(&gray,siz);
		imageReplace(gray,outimage);
	}

	if(type == PROC_BW)
	{
		IplImage* gray = NULL;//�Ҷ�ͼ��
		rgb2gray(inimage,&gray);//תΪ�Ҷ�ͼ��

		//Ȼ���ֵ��
		IplImage *bwimage = cvCreateImage(cvGetSize(gray), 8, 1);//����ռ�
		ThresholdOtsu(gray,bwimage,CV_THRESH_BINARY);	//�Զ���ֵ��ֵ��

		cropimage(bwimage,outimage,CV_THRESH_BINARY);//�ҵ������ĺڱߣ����г���
		imageResize(outimage,siz);//������ͳһ��С

		IplImage* bwimgcrop = cvCreateImage(cvGetSize(*outimage), 8, 1);
		ThresholdOtsu(*outimage,bwimgcrop,CV_THRESH_BINARY);//���������ֵ������Ϊresize�󣬻��ֵ��������ز�ȫ����255
		imageReplace(bwimgcrop,outimage);

		cvReleaseImage(&gray);;
		cvReleaseImage(&bwimage);

		/*cvNamedWindow("��ֵ��");
		cvShowImage("��ֵ��",*outimage);*/
	}


	if(type == PROC_EDGE)
	{
		IplImage* gray = NULL;//�Ҷ�ͼ��
		rgb2gray(inimage,&gray);//תΪ�Ҷ�ͼ��

		Mat grayMat  = gray;
		Mat sketchimg = sketch(grayMat);//����
		threshold(sketchimg, sketchimg,250, 255, CV_THRESH_BINARY);//����֮���ֵ��

		IplImage sketchimage = sketchimg;
		
		cropimage(&sketchimage,outimage,CV_THRESH_BINARY);//�ҵ������ĺڱߣ����г���
		imageResize(outimage,siz);//������ͳһ��С

		IplImage* bwimgcrop = cvCreateImage(cvGetSize(*outimage), 8, 1);
		ThresholdOtsu(*outimage,bwimgcrop,CV_THRESH_BINARY);//���������ֵ������Ϊresize�󣬻��ֵ��������ز�ȫ����255
		imageReplace(bwimgcrop,outimage);


		cvReleaseImage(&gray);;
		

		/*cvNamedWindow("��Ե");
		imshow("��Ե",sketchimg);*/
	}

	IplImage* gray = NULL;//�Ҷ�ͼ��
	rgb2gray(inimage,&gray);//תΪ�Ҷ�ͼ��
	

	//��һ����С
	IplImage* size_img = cvCreateImage( siz,8,1);//�趨ͬһ�ߴ�
	cvResize( gray, size_img, CV_INTER_LINEAR );//���Ҷ�ͼ������С��ͬһ�ߴ�
	
	if(type == PROC_BW)
	{
		//Ȼ���ֵ��
		IplImage *bwimage = cvCreateImage(cvGetSize(size_img), 8, 1);//����ռ�
		ThresholdOtsu(size_img,bwimage,CV_THRESH_BINARY);	//�Զ���ֵ��ֵ��

		

		*outimage = cvCloneImage(bwimage);

		cvReleaseImage(&gray);
		cvReleaseImage(&size_img);
		cvReleaseImage(&bwimage);
		return;
	}
}

//��������õ�����RECT����
//shape ���������������
//organimage Ҫ�õ�������ͼ��
//organRect �õ�������ͼ��λ��Rect
//expandrate ��չϵ��
void getOrganRect(asm_shape shape,vector<int> indexNumber, CvRect &organRect,double expandrate=0.1)
{
	int pointnum = indexNumber.size();//��������
	vector<int> xs;//���x����
	vector<int> ys;//���y����

	int i=0;
	for(i=0;i<pointnum;i++)//����ÿ����������
	{
		int index = indexNumber[i];//��Ӧ������

		int x = cvRound(shape[index].x);
		int y = cvRound(shape[index].y);
				
		CvPoint point;
		point.x = x;
		point.y = y;

		xs.push_back(x);//x����
		ys.push_back(y);//y����
	}

	int minx = *min_element(xs.begin(),xs.end());  //����Сֵ
	int maxx = *max_element(xs.begin(),xs.end());  //�����ֵ

	int miny = *min_element(ys.begin(),ys.end());  //����Сֵ
	int maxy = *max_element(ys.begin(),ys.end());  //�����ֵ

	//Ϊ�˸������Ľ�ȡ��Ҫʹ��ȡ������΢��һЩ
	int height = maxy - miny + 1;
	int width = maxx - minx + 1;
	int diffheight = height * expandrate;
	int diffwidth = width * expandrate;

	//��֯rect����
	organRect.x = minx - diffwidth;
	organRect.y = miny - diffheight;
	organRect.width = width + 2*diffwidth;
	organRect.height = height + 2*diffheight;
}

//���������ȡ����ͼ�����򣬵õ�����ͼ��
//image ԭʼͼ��
//indexNumber ���ٶ�Ӧ�ĵ㼯�����
//shape ���������������
//organimage Ҫ�õ�������ͼ��
//organRect �õ�������ͼ��λ��Rect
//points ���ٵ�λ�ü���
void getOrganImage(IplImage *image,asm_shape shape,vector<int> indexNumber, 
	IplImage **organimage,CvRect &organRect,vector<CvPoint> &points,double expandrate=0.1)
{
	int pointnum = indexNumber.size();//��������
	vector<int> xs;//���x����
	vector<int> ys;//���y����

	int i=0;
	for(i=0;i<pointnum;i++)//����ÿ����������
	{
		int index = indexNumber[i];//��Ӧ������

		int x = cvRound(shape[index].x);
		int y = cvRound(shape[index].y);
				
		CvPoint point;
		point.x = x;
		point.y = y;

		xs.push_back(x);//x����
		ys.push_back(y);//y����
		points.push_back(point);//�㼯��
	}

	int minx = *min_element(xs.begin(),xs.end());  //����Сֵ
	int maxx = *max_element(xs.begin(),xs.end());  //�����ֵ

	int miny = *min_element(ys.begin(),ys.end());  //����Сֵ
	int maxy = *max_element(ys.begin(),ys.end());  //�����ֵ

	//Ϊ�˸������Ľ�ȡ��Ҫʹ��ȡ������΢��һЩ
	int height = maxy - miny + 1;
	int width = maxx - minx + 1;
	int diffheight = height * expandrate;
	int diffwidth = width * expandrate;

	//��֯rect����
	organRect.x = minx - diffwidth;
	organRect.y = miny - diffheight;
	organRect.width = width + 2*diffwidth;
	organRect.height = height + 2*diffheight;

	//��ȡ����Ӧ��ͼ������
	cvSetImageROI(image,organRect);//�����ȵ��������ڸ���
	*organimage = cvCreateImage( cvSize(organRect.width,organRect.height),IPL_DEPTH_8U, image->nChannels );
	cvCopy(image,*organimage);//��Ϊ�趨���ȵ���������ֻ�Ḵ����ز���
	cvResetImageROI(image);//ȡ���ȵ�
}
