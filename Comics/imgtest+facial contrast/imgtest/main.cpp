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
	PROC_GRAY,//灰度
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


//预处理,主要是转为灰度，二值化，大小归一等等
void preproc(IplImage* inimage,IplImage** outimage,CvSize siz,PROCTYPE type)
{
	
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

//这个函数获取器官图像区域
//image 原始图像
//indexNumber 器官对应的点集合序号
//shape 检测出的特征点对象
//organimage 要得到的器官图像
//organRect 得到的器官图像位置Rect
//points 器官点位置集合
void getOrganImage(IplImage *image,asm_shape shape,vector<int> indexNumber, 
	IplImage **organimage,CvRect &organRect,vector<CvPoint> &points)
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
	int diffheight = height * 0.1;
	int diffwidth = width * 0.1;

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


void getPCA(const char* strname,int num,PCA &pcaObject,Mat &alltraindataPCA)
{
	int i;
	char strfilename[256] = {0};

	Mat alltraindata = Mat();

	for(i=1;i<=num;i++)
	{
		sprintf(strfilename,"五官素材\\%s\\%s%d.png",strname,strname,i);
		IplImage * image = cvLoadImage(strfilename);
		IplImage* imageproc = NULL;//预处理图像
		preproc(image,&imageproc,cvSize(60,20),PROC_BW);//图像计算特征之前先预处理下

		
		//为了计算pca，将样本处理下，每个图像样本转为一行的矩阵
		//Mat img(imageproc);//转为Mat格式
		Mat img = cv::cvarrToMat(imageproc);
		Mat doubleimg;//为了数据更好的处理，转为0-1之间浮点数
		img.convertTo(doubleimg,CV_64F,double(1) /double(255),0);
		Mat rowimg = doubleimg.reshape(0,1);//拉伸为一行的矩阵
		alltraindata.push_back(rowimg);//这一行存储到picDatas内

		pcaObject(alltraindata,Mat(), CV_PCA_DATA_AS_ROW,30);//先得到PCA参数，存在pcaObject内
		alltraindataPCA = pcaObject.project(alltraindata);//将原始数据通过PCA方向投影,降维后的数据为alltraindataPCA

		cvReleaseImage(&imageproc);
	}
}

//计算欧式距离
double Eucdiatance(Mat feature1,Mat feature2)
{
	double dist = 0;

	Mat diff = feature1 - feature2;//二者相减
	int rows = diff.rows;
	int cols = diff.cols;

	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			dist = dist + pow(diff.at<double>(i,j),2);//二者对应维度的值相减，平方和
		}
	}
	

	return dist;
}

int main()
{
	asmfitting fit_asm;//特征点检测模型
	char* model_name = "my68-1d.amf";//模型文件
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//人脸定位需要的xml
	char* strfilename = "5.jpg";//要读取测试的图像文件名

	PCA pcaObject;//PCA对象
	Mat alltraindataPCA = Mat();//PCA降维后的数据
	int samplenum = 6;
	getPCA("左眼",samplenum,pcaObject,alltraindataPCA);//训练pca

	if(fit_asm.Read(model_name) == false)//装载特征点模型
	{
		printf("特征点模型装载失败\r\n");
		return -1;
	}
	
	if(init_detect_cascade(cascade_name) == false)//装载人脸检测xml
	{
		printf("人脸检测模型装载失败\r\n");
		return -1;
	}
	
	IplImage * image = cvLoadImage(strfilename);
	if(image == 0)
	{
		printf("打开图像失败: %s\r\n", strfilename);
		return -1;
	}

	asm_shape shape, detshape;
	bool flag =detect_one_face(detshape, image);//从图中寻找人脸，寻找最靠中心的的人脸

	if(!flag) 
	{	
		printf("没有检测到人脸\r\n");
		return -1;
	}



	//初始化寻找特征点
	InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	fit_asm.Fitting(shape, image, 20);//迭代寻找特征点，使得更准确

	//找完特征点，后截取出主要的器官出来
	int i,j;
	//寻找左眼区域
	vector<CvPoint> lefteyePoints;//左眼点集合
	CvRect lefteyeRect;//左眼rect区域
	IplImage * lefteyeimage = NULL;
	vector<int> lefteyeIndexs;
	for(i=27;i<=31;i++)
	{
		lefteyeIndexs.push_back(i);
	}
	getOrganImage(image,shape,lefteyeIndexs, &lefteyeimage,lefteyeRect,lefteyePoints);//寻找器官出来


	//然后二值化
	IplImage *bwlefteyeimage = NULL;//分配空间
	preproc(lefteyeimage,&bwlefteyeimage,cvSize(60,20),PROC_BW);//图像计算特征之前先预处理下

	//寻找素材库里最接近的
	//Mat img(bwlefteyeimage);//转为Mat格式
	Mat img = cv::cvarrToMat(bwlefteyeimage);
	Mat doubleimg;//为了数据更好的处理，转为0-1之间浮点数
	img.convertTo(doubleimg,CV_64F,double(1) /double(255),0);
	Mat rowimg = doubleimg.reshape(0,1);//拉伸为一行的矩阵
	
	Mat imgPCA = pcaObject.project(rowimg);//将原始数据通过PCA方向投影,降维后的数据为imgPCA
			
	int alltrainpicnum = alltraindataPCA.rows;//所有训练样本个数
	double minvalue =DBL_MAX;
	int mintype = -1;//距离最小的，所对应的种类
	double dist = 0;
	for(int i=0;i<alltrainpicnum;i++)//遍历所有图
	{
		Mat tmp = alltraindataPCA.row(i);//取出一行数据，也就是一个图的降维后的数据
		double dist = Eucdiatance(imgPCA,tmp);//计算欧氏距离
		if(dist < minvalue)//找出最小的距离，以及所在索引
		{
			minvalue = dist;
			mintype = i+1;//从类型列表里取出对应的类型
		}
	}

	char strname[256] = {0};
	sprintf(strname,"五官素材\\左眼\\左眼%d.png",mintype);
	IplImage *selectimage = cvLoadImage(strname);
	
	printf("%s\r\n",strname);

	//显示特征点图像
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3, 0, 1, 8);
	int keypointsnum = shape.NPoints();//关键点数
	for (j=0;j<keypointsnum;j++)
	{
		float x = shape[j].x;
		float y = shape[j].y;
				
		CvPoint centerpoint;
		centerpoint.x = cvRound(x);
		centerpoint.y = cvRound(y);
		cvCircle( image, centerpoint ,1 , CV_RGB(255,0,0),1 );		
	}

	cvNamedWindow("特征点", 0);
	cvShowImage("特征点", image);	

	cvNamedWindow("左眼", 1);
	cvShowImage("左眼", bwlefteyeimage);	

	cvNamedWindow("左眼选择", 1);
	cvShowImage("左眼选择", selectimage);	

	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}