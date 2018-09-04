#include "mymethod.h"




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
	asmfitting fit_asm;//特征点检测模型
	char* model_name = "my68-1d.amf";//模型文件
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//人脸定位需要的xml
	char* strfilename = "男1.jpg";//要读取测试的图像文件名


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
	fit_asm.Fitting(shape, image, 20);//迭代寻找特征点，使得更准确

	//找完特征点，后截取出主要的器官出来
	int i,j;

	//寻找匹配左眼区域
	vector<Point> lefteyePoints;//左眼点集合
	CvRect lefteyeRect;//左眼rect区域
	IplImage * lefteyeimage = NULL;
	vector<int> lefteyeIndexs;
	for(i=27;i<=30;i++)
	{
		lefteyeIndexs.push_back(i);
	
	}
	getOrganImage(image,shape,lefteyeIndexs,&lefteyeimage,lefteyeRect,lefteyePoints);
	contours.push_back(lefteyePoints);

	//寻找匹配右眼区域
	vector<Point> righteyePoints;//左眼点集合
	CvRect righteyeRect;//左眼rect区域
	IplImage * righteyeimage = NULL;
	vector<int> righteyeIndexs;
	for(i=32;i<=35;i++)
	{
		righteyeIndexs.push_back(i);
	}
	getOrganImage(image,shape,righteyeIndexs,&righteyeimage,righteyeRect,righteyePoints);
	contours.push_back(righteyePoints);

	//寻找匹配左眉区域
	vector<Point> leftbrowPoints;//左眉毛点集合
	CvRect leftbrowRect;//左眉毛rect区域
	IplImage * leftbrowimage = NULL;
	vector<int> leftbrowIndexs;
	for(i=21;i<=26;i++)
	{
		leftbrowIndexs.push_back(i);
	}
	getOrganImage(image,shape,leftbrowIndexs,&leftbrowimage,leftbrowRect,leftbrowPoints);
	contours.push_back(leftbrowPoints);


	//寻找匹配右眉毛区域
	vector<Point> rightbrowPoints;//右眉点集合
	CvRect rightbrowRect;//右眉rect区域
	IplImage * rightbrowimage = NULL;
	vector<int> rightbrowIndexs;
	for(i=15;i<=20;i++)
	{
		rightbrowIndexs.push_back(i);
	}
	getOrganImage(image,shape,rightbrowIndexs,&rightbrowimage,rightbrowRect,rightbrowPoints);
	contours.push_back(rightbrowPoints);


	//IplImage *grayimg = NULL;
	//rgb2gray(rightbrowimage,&grayimg);
	//IplImage *bwimage = cvCreateImage(cvGetSize(rightbrowimage), 8, 1);//分配空间
	//	ThresholdOtsu(grayimg,bwimage,CV_THRESH_BINARY);	//自动阈值二值化


	//寻找匹配嘴型区域
	vector<Point> mouthPoints;//右眉点集合
	CvRect mouthRect;//右眉rect区域
	IplImage * mouthimage = NULL;
	vector<int> mouthIndexs;
	for(i=48;i<=59;i++)
	{
		mouthIndexs.push_back(i);
	}
	mouthIndexs.push_back(48);
	mouthIndexs.push_back(60);
	mouthIndexs.push_back(61);
	mouthIndexs.push_back(62);
	mouthIndexs.push_back(54);
	mouthIndexs.push_back(63);
	mouthIndexs.push_back(64);
	mouthIndexs.push_back(65);

	getOrganImage(image,shape,mouthIndexs,&mouthimage,mouthRect,mouthPoints);
	contours.push_back(mouthPoints);

	//寻找匹配鼻型区域
	vector<Point> nosePoints;//鼻子点集合
	CvRect noseRect;//鼻子rect区域
	IplImage * noseimage = NULL;
	vector<int> noseIndexs;
	for(i=37;i<=45;i++)
	{
		noseIndexs.push_back(i);
	}
	/*noseIndexs.push_back(47);
	noseIndexs.push_back(46);*/
	getOrganImage(image,shape,noseIndexs,&noseimage,noseRect,nosePoints);
	contours.push_back(nosePoints);

	vector<Point> nosePoints2;//鼻子点集合
	CvRect noseRect2;//鼻子rect区域
	IplImage * noseimage2 = NULL;
	vector<int> noseIndexs2;
	noseIndexs2.push_back(41);
	noseIndexs2.push_back(46);
	noseIndexs2.push_back(67);
	noseIndexs2.push_back(47);

	/*noseIndexs.push_back(47);
	noseIndexs.push_back(46);*/
	getOrganImage(image,shape,noseIndexs2,&noseimage2,noseRect2,nosePoints2);
	contours.push_back(nosePoints2);

	vector<Point> facePoints;//脸型点集合
	CvRect faceRect;//鼻子rect区域
	IplImage * faceimage = NULL;
	vector<int> faceIndexs;
	for(i=0;i<=14;i++)
	{
		faceIndexs.push_back(i);
	}
	getOrganImage(image,shape,faceIndexs,&faceimage,faceRect,facePoints);
	contours.push_back(facePoints);

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

	Mat flatimg(image->height,image->width,CV_8UC1,Scalar::all(255));//生成一个全白色的图像
	drawContours(flatimg, contours, 0, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 1, CV_RGB(0,0,0),2, 8);
	drawContours(flatimg, contours, 2, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 3, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 4, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 5, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 6, CV_RGB(0,0,0), 2, 8);

	cvNamedWindow("特征点", 0);
	cvShowImage("特征点", image);	

	cvNamedWindow("左眼", 1);
	cvShowImage("左眼", lefteyeimage);	

	cvNamedWindow("右眼", 1);
	cvShowImage("右眼", righteyeimage);	

	cvNamedWindow("左眉", 1);
	cvShowImage("左眉", leftbrowimage);	

	cvNamedWindow("右眉", 1);
	cvShowImage("右眉", rightbrowimage);	

	
	cvNamedWindow("鼻子", 1);
	cvShowImage("鼻子", noseimage);	

	cvNamedWindow("嘴巴", 1);
	cvShowImage("嘴巴", mouthimage);	

	cvNamedWindow("轮廓", 0);
	imshow("轮廓", flatimg);	

	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}