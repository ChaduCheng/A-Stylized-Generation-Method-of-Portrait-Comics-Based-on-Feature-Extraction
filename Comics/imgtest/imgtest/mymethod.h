
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


enum PROCTYPE//处理的类型
{
	PROC_ORI,//原始
	PROC_GRAY,//灰度化
	PROC_BW,//二值化
	PROC_EDGE//边缘
};

int imageResize(IplImage** pi,CvSize sz)
{
	IplImage *src = *pi;
	if((src->width == sz.width) && (src->height == sz.height)) //如果尺寸本来就想等，不必做任何变动
	{
		return 0;
	}
	IplImage *dst = cvCreateImage(sz,src->depth,src->nChannels);//定义目标大小
	cvResize(*pi,dst);//调整尺寸

	cvReleaseImage(pi);     
	*pi = dst;
	return (1); 
}

int  imageReplace(IplImage* pi,IplImage** ppo)  //  IplImage替换
{
	if (*ppo) 
		cvReleaseImage(ppo);                //  释放原来位图
	(*ppo) = pi;                            //  位图换名
	return(1);
}

int  imageClone(IplImage* pi,IplImage** ppo)  //  复制 IplImage 位图
{
	if (*ppo) {
		cvReleaseImage(ppo);                //  释放原来位图
	}
	(*ppo) = cvCloneImage(pi);              //  复制新位图
	return(1);
}


void rgb2gray(IplImage* pi,IplImage** ppo)
{
	if (pi->nChannels == 1)//如果已经是灰度图了
	{
		imageClone(pi,ppo);//直接复制
	}
	else
	{
		if (*ppo) 
			cvReleaseImage(ppo);                //  释放原来位图
		*ppo = cvCreateImage(cvGetSize(pi), 8, 1);
		cvCvtColor(pi,*ppo,CV_BGR2GRAY);//rgb转为灰度图
	}
}

//OTSU自动二值化
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

//这个函数，剪切图像，使其紧紧包含，可以找寻紧贴黑点的，也可以找紧贴白点的
void cropimage(IplImage* inimage,IplImage** outimage,int type)
{
	IplImage* gray = NULL;//灰度图像
	rgb2gray(inimage,&gray);//转为灰度图像

	//然后二值化
	IplImage *bwimage = cvCreateImage(cvGetSize(gray), 8, 1);//分配空间
	ThresholdOtsu(gray,bwimage,type);	//自动阈值二值化

	int left,right,top,bottom;
	//左边界
	for (int i=0;i<bwimage->width;i++)//遍历每一列
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->height;j++)//遍历每一行
		{
			double value = cvGetReal2D(bwimage,j,i);//获取像素值
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//找到黑点列了
		{
			left = i;//左边界
			break;
		}

	}

	//右边界
	for (int i=bwimage->width-1; i>=0; i--)//遍历每一列
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->height;j++)//遍历每一行
		{
			double value = cvGetReal2D(bwimage,j,i);//获取像素值
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//找到黑点列了
		{
			right = i;//右边界
			break;
		}
	}

	//上边界
	for (int i=0;i<bwimage->height;i++)//遍历每一行
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->width;j++)//遍历每一列
		{
			double value = cvGetReal2D(bwimage,i,j);//获取像素值
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//找到黑点列了
		{
			top = i;//左边界
			break;
		}

	}

	//下边界
	for (int i=bwimage->height-1; i>=0; i--)//遍历每一行
	{	
		bool bfind = false;
		for(int j=0;j<bwimage->width;j++)//遍历每一列
		{
			double value = cvGetReal2D(bwimage,i,j);//获取像素值
			if(value < 0.1)
			{
				bfind = true;
				break;
			}
		}
		if(bfind)//找到黑点列了
		{
			bottom = i;//下边界
			break;
		}
	}

	CvRect r;//组织矩形区域
	r.x = left;
	r.y = top;
	r.height = bottom-top+1;
	r.width = right-left+1;//
	cvSetImageROI(inimage,r);//在图中选出区域,截取出来
	*outimage = cvCreateImage( cvGetSize(inimage),8, inimage->nChannels);
	cvCopy(inimage,*outimage);
	cvResetImageROI(inimage);

	cvReleaseImage(&gray);
	cvReleaseImage(&bwimage);
}

//这个函数实现素描
Mat sketch(Mat src)
{
	Mat dst = src.clone();
	int width=src.cols;  
	int heigh=src.rows;  

	Mat gray1;
	//反色  
	addWeighted(src,-1,NULL,0,255,gray1);  
	//高斯模糊,高斯核的Size与最后的效果有关  
	GaussianBlur(gray1,gray1,Size(11,11),0);  
	//融合：颜色减淡  
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

//预处理,主要是转为灰度，二值化，大小归一等等
void preproc(IplImage* inimage,IplImage** outimage,CvSize siz,PROCTYPE type)
{
	
	if(type == PROC_ORI)
	{
		cropimage(inimage,outimage,CV_THRESH_BINARY);//找到紧贴的黑边，剪切出来
		imageResize(outimage,siz);//调整到统一大小
	}
	if(type == PROC_GRAY)
	{
		IplImage* gray = NULL;//灰度图像
		rgb2gray(inimage,&gray);//转为灰度图像
		imageResize(&gray,siz);
		imageReplace(gray,outimage);
	}

	if(type == PROC_BW)
	{
		IplImage* gray = NULL;//灰度图像
		rgb2gray(inimage,&gray);//转为灰度图像

		//然后二值化
		IplImage *bwimage = cvCreateImage(cvGetSize(gray), 8, 1);//分配空间
		ThresholdOtsu(gray,bwimage,CV_THRESH_BINARY);	//自动阈值二值化

		cropimage(bwimage,outimage,CV_THRESH_BINARY);//找到紧贴的黑边，剪切出来
		imageResize(outimage,siz);//调整到统一大小

		IplImage* bwimgcrop = cvCreateImage(cvGetSize(*outimage), 8, 1);
		ThresholdOtsu(*outimage,bwimgcrop,CV_THRESH_BINARY);//这里继续二值化，因为resize后，会插值，造成像素不全等于255
		imageReplace(bwimgcrop,outimage);

		cvReleaseImage(&gray);;
		cvReleaseImage(&bwimage);

		/*cvNamedWindow("二值化");
		cvShowImage("二值化",*outimage);*/
	}


	if(type == PROC_EDGE)
	{
		IplImage* gray = NULL;//灰度图像
		rgb2gray(inimage,&gray);//转为灰度图像

		Mat grayMat  = gray;
		Mat sketchimg = sketch(grayMat);//素描
		threshold(sketchimg, sketchimg,250, 255, CV_THRESH_BINARY);//素描之后二值化

		IplImage sketchimage = sketchimg;
		
		cropimage(&sketchimage,outimage,CV_THRESH_BINARY);//找到紧贴的黑边，剪切出来
		imageResize(outimage,siz);//调整到统一大小

		IplImage* bwimgcrop = cvCreateImage(cvGetSize(*outimage), 8, 1);
		ThresholdOtsu(*outimage,bwimgcrop,CV_THRESH_BINARY);//这里继续二值化，因为resize后，会插值，造成像素不全等于255
		imageReplace(bwimgcrop,outimage);


		cvReleaseImage(&gray);;
		

		/*cvNamedWindow("边缘");
		imshow("边缘",sketchimg);*/
	}

	IplImage* gray = NULL;//灰度图像
	rgb2gray(inimage,&gray);//转为灰度图像
	

	//归一化大小
	IplImage* size_img = cvCreateImage( siz,8,1);//设定同一尺寸
	cvResize( gray, size_img, CV_INTER_LINEAR );//将灰度图调整大小到同一尺寸
	
	if(type == PROC_BW)
	{
		//然后二值化
		IplImage *bwimage = cvCreateImage(cvGetSize(size_img), 8, 1);//分配空间
		ThresholdOtsu(size_img,bwimage,CV_THRESH_BINARY);	//自动阈值二值化

		

		*outimage = cvCloneImage(bwimage);

		cvReleaseImage(&gray);
		cvReleaseImage(&size_img);
		cvReleaseImage(&bwimage);
		return;
	}
}

//这个函数得到器官RECT区域
//shape 检测出的特征点对象
//organimage 要得到的器官图像
//organRect 得到的器官图像位置Rect
//expandrate 扩展系数
void getOrganRect(asm_shape shape,vector<int> indexNumber, CvRect &organRect,double expandrate=0.1)
{
	int pointnum = indexNumber.size();//特征点数
	vector<int> xs;//点的x集合
	vector<int> ys;//点的y集合

	int i=0;
	for(i=0;i<pointnum;i++)//遍历每个点找最大的
	{
		int index = indexNumber[i];//对应的索引

		int x = cvRound(shape[index].x);
		int y = cvRound(shape[index].y);
				
		CvPoint point;
		point.x = x;
		point.y = y;

		xs.push_back(x);//x集合
		ys.push_back(y);//y集合
	}

	int minx = *min_element(xs.begin(),xs.end());  //求最小值
	int maxx = *max_element(xs.begin(),xs.end());  //求最大值

	int miny = *min_element(ys.begin(),ys.end());  //求最小值
	int maxy = *max_element(ys.begin(),ys.end());  //求最大值

	//为了更完整的截取，要使截取区域稍微大一些
	int height = maxy - miny + 1;
	int width = maxx - minx + 1;
	int diffheight = height * expandrate;
	int diffwidth = width * expandrate;

	//组织rect区域
	organRect.x = minx - diffwidth;
	organRect.y = miny - diffheight;
	organRect.width = width + 2*diffwidth;
	organRect.height = height + 2*diffheight;
}

//这个函数获取器官图像区域，得到器官图像
//image 原始图像
//indexNumber 器官对应的点集合序号
//shape 检测出的特征点对象
//organimage 要得到的器官图像
//organRect 得到的器官图像位置Rect
//points 器官点位置集合
void getOrganImage(IplImage *image,asm_shape shape,vector<int> indexNumber, 
	IplImage **organimage,CvRect &organRect,vector<CvPoint> &points,double expandrate=0.1)
{
	int pointnum = indexNumber.size();//特征点数
	vector<int> xs;//点的x集合
	vector<int> ys;//点的y集合

	int i=0;
	for(i=0;i<pointnum;i++)//遍历每个点找最大的
	{
		int index = indexNumber[i];//对应的索引

		int x = cvRound(shape[index].x);
		int y = cvRound(shape[index].y);
				
		CvPoint point;
		point.x = x;
		point.y = y;

		xs.push_back(x);//x集合
		ys.push_back(y);//y集合
		points.push_back(point);//点集合
	}

	int minx = *min_element(xs.begin(),xs.end());  //求最小值
	int maxx = *max_element(xs.begin(),xs.end());  //求最大值

	int miny = *min_element(ys.begin(),ys.end());  //求最小值
	int maxy = *max_element(ys.begin(),ys.end());  //求最大值

	//为了更完整的截取，要使截取区域稍微大一些
	int height = maxy - miny + 1;
	int width = maxx - minx + 1;
	int diffheight = height * expandrate;
	int diffwidth = width * expandrate;

	//组织rect区域
	organRect.x = minx - diffwidth;
	organRect.y = miny - diffheight;
	organRect.width = width + 2*diffwidth;
	organRect.height = height + 2*diffheight;

	//截取出对应的图像块出来
	cvSetImageROI(image,organRect);//设置热点区域，用于复制
	*organimage = cvCreateImage( cvSize(organRect.width,organRect.height),IPL_DEPTH_8U, image->nChannels );
	cvCopy(image,*organimage);//因为设定了热点区域，所以只会复制相关部分
	cvResetImageROI(image);//取消热点
}
