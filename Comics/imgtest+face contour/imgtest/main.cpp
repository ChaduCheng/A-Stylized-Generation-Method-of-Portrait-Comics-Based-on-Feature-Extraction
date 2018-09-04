#include "mymethod.h"
#include<iostream>
#include<time.h>



void getPCA(const char* strname,int num,PCA &pcaObject,Mat &alltraindataPCA,CvSize siz)
{
	int i;
	char strfilename[256] = {0};

	Mat alltraindata = Mat();

	for(i=1;i<=num;i++)
	{
		sprintf(strfilename,"五官素材\\%s\\%s%d.png",strname,strname,i);
		IplImage * image = cvLoadImage(strfilename);
		IplImage* imageproc = NULL;//预处理图像
		preproc(image,&imageproc,siz,PROC_BW);//图像计算特征之前先预处理下

		
		//为了计算pca，将样本处理下，每个图像样本转为一行的矩阵
		Mat img(imageproc);//转为Mat格式
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

//这个函数从素材库选取一个最合适的出来
//image 原始图像
//indexNumber 器官对应的点集合序号
//shape 检测出的特征点对象
//siz 调整到统一大小
//PROCTYPE type 特征类型 BW二值化  EDGE边缘素描
//strtypename 器官名称
//strfilename 挑选出的图片文件名
void selectOrganImage(IplImage *image,asm_shape shape,vector<int> indexNumber,CvSize siz,PROCTYPE proctype,const char*strtypename,
	IplImage **organimage,CvRect &organRect,char *strfilename)
{
	PCA pcaObject;//PCA对象
	Mat alltraindataPCA = Mat();//PCA降维后的数据
	int samplenum = 6;
	getPCA(strtypename,samplenum,pcaObject,alltraindataPCA,siz);//训练pca

	//依据特征点序号，找出器官出来
	int i,j;
	//寻找器官区域
	vector<Point> organPoints;//特征点集合
	
	getOrganImage(image,shape,indexNumber, organimage,organRect,organPoints);//寻找器官出来

	//然后预处理
	IplImage *bworganimage = NULL;//分配空间
	preproc(*organimage,&bworganimage,siz,proctype);//图像计算特征之前先预处理下

	//寻找素材库里最接近的
	Mat img(bworganimage);//转为Mat格式
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

	sprintf(strfilename,"五官素材\\%s\\%s%d.png",strtypename,strtypename,mintype);

}

int main()
{
	clock_t start_time=clock();
	asmfitting fit_asm;//特征点检测模型
	char* model_name = "my68-1d.amf";//模型文件
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//人脸定位需要的xml
	char* strfilename = "男3.jpg";//要读取测试的图像文件名


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

	vector<vector<Point>> contours;

	//初始化寻找特征点
	InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	fit_asm.Fitting(shape, image, 10);//迭代寻找特征点，使得更准确
	shape[9].x = shape[9].x -9;
	shape[10].x = shape[10].x -5;
	clock_t end_time=clock();
    cout<< "人脸五官定位时间: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
	//生成一个白板图像，用于描述轮廓图
	IplImage *flatimg = cvCreateImage(cvGetSize(image),8,1);
	cvSet(flatimg,CV_RGB(255,255,255));

	int i,j;
	CvPoint point1;
	CvPoint point2;
	//绘制左眼
	for(i=27;i<30;i++)
	{
		//点1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//点2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//绘制闭合的曲线，尾首链接
	int x = cvRound(shape[27].x);
	int y = cvRound(shape[27].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);

	//绘制右眼
	for(i=32;i<35;i++)
	{
		//点1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//点2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//绘制闭合的曲线，尾首链接
	x = cvRound(shape[32].x);
	y = cvRound(shape[32].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);


	//绘制左眉毛
	for(i=21;i<26;i++)
	{
		//点1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//点2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//绘制闭合的曲线，尾首链接
	x = cvRound(shape[21].x);
	y = cvRound(shape[21].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);

	//绘制右眉毛
	for(i=15;i<20;i++)
	{
		//点1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//点2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//绘制闭合的曲线，尾首链接
	x = cvRound(shape[15].x);
	y = cvRound(shape[15].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);

	//绘制鼻子
	vector<int> noseIndexs;//找出需要绘制的顺序
	for(i=37;i<=40;i++)
	{
		noseIndexs.push_back(i);
	}
	noseIndexs.push_back(46);
	noseIndexs.push_back(41);
	noseIndexs.push_back(47);
	for(i=42;i<=45;i++)
	{
		noseIndexs.push_back(i);
	}
	//绘制鼻子轮廓
	for(i=0;i<noseIndexs.size()-1;i++)
	{
		int index = noseIndexs[i];
		//点1
		int x = cvRound(shape[index].x);
		int y = cvRound(shape[index].y);	
		
		point1.x = x;
		point1.y = y;

		index = noseIndexs[i+1];//下一个点
		//点2
		x = cvRound(shape[index].x);
		y = cvRound(shape[index].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}


	//绘制嘴巴轮廓
	for(i=48;i<59;i++)
	{
		//点1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//点2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//绘制闭合的曲线，尾首链接
	x = cvRound(shape[48].x);
	y = cvRound(shape[48].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);


		//绘制右眼
	for(i=0;i<14;i++)
	{
		//点1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//点2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}

	cvNamedWindow("轮廓图",1);
	cvShowImage("轮廓图",flatimg);


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
		cvCircle( image, centerpoint ,2 , CV_RGB(255,0,0),-1 );		
	}
	cvNamedWindow("特征点",1);
	cvShowImage("特征点",image);



	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}