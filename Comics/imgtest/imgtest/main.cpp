#include "mymethod.h"

#include <stdlib.h>


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

void getPCA2(const char* strname,int num,PCA &pcaObject,Mat &alltraindataPCA,CvSize siz)
{
	int i;
	char strfilename[256] = {0};

	Mat alltraindata = Mat();

	for(i=1;i<=num;i++)
	{
		sprintf(strfilename,"五官素材\\%s训练\\%s%d.png",strname,strname,i);
		IplImage * image = cvLoadImage(strfilename);
		IplImage* imageproc = NULL;//预处理图像
		preproc(image,&imageproc,siz,PROC_GRAY);//图像计算特征之前先预处理下

		
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
	IplImage **organimage,IplImage **organimageproc,CvRect &organRect,char *strfilename)
{
	PCA pcaObject;//PCA对象
	Mat alltraindataPCA = Mat();//PCA降维后的数据
	int samplenum = 6;
	getPCA(strtypename,samplenum,pcaObject,alltraindataPCA,siz);//训练pca

	//依据特征点序号，找出器官出来
	int i,j;
	//寻找器官区域
	vector<CvPoint> organPoints;//特征点集合
	
	getOrganImage(image,shape,indexNumber, organimage,organRect,organPoints);//寻找器官出来

	//然后预处理
	preproc(*organimage,organimageproc,siz,proctype);//图像计算特征之前先预处理下

	//寻找素材库里最接近的
	Mat img(*organimageproc);//转为Mat格式
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

//这个函数从素材库选取一个最合适的出来
//image 原始图像
//indexNumber 器官对应的点集合序号
//shape 检测出的特征点对象
//siz 调整到统一大小
//PROCTYPE type 特征类型 BW二值化  EDGE边缘素描
//strtypename 器官名称
//strfilename 挑选出的图片文件名
void selectOrganImage2(IplImage *image,asm_shape shape,vector<int> indexNumber,CvSize siz,const char*strtypename,int samplenum,
	IplImage **organimage,IplImage **organimageproc,CvRect &organRect,char *strfilename)
{
	PCA pcaObject;//PCA对象
	Mat alltraindataPCA = Mat();//PCA降维后的数据
	getPCA2(strtypename,samplenum,pcaObject,alltraindataPCA,siz);//训练pca

	//依据特征点序号，找出器官出来
	int i,j;
	//寻找器官区域
	vector<CvPoint> organPoints;//特征点集合
	
	getOrganImage(image,shape,indexNumber, organimage,organRect,organPoints);//寻找器官出来

	//然后预处理
	preproc(*organimage,organimageproc,siz,PROC_GRAY);//图像计算特征之前先预处理下

	//寻找素材库里最接近的
	Mat img(*organimageproc);//转为Mat格式
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

//这个函数计算出新的Rect区域
//targetx1 targetx2 脸轮廓最高两个点的x坐标
//targety1 脸轮廓的高点y坐标
//targety2 脸轮廓的低点y坐标
//organRect 原始器官区域
//neworganRect 得到的要粘贴的器官区域
void computeNewRect(asm_shape shape,int targetx1,int targetx2,int targety1,int targety2,CvRect organRect,CvRect &neworganRect)
{
	//要粘贴的目标脸的高度宽度
	int targetwidth = targetx2 - targetx1 +1;//脸轮廓的宽度
	int targetheight = targety2 - targety1 + 1;//脸轮廓的高度

	//原始脸的宽度和高度计算下
	int x1 = shape[0].x;
	int x2 = shape[14].x;
	int width = x2-x1+1;
	int y1 = shape[0].y;
	int y2 = shape[7].y;
	int height = y2-y1+1;

	//计算比例
	double wrate = double(targetwidth)/double(width);
	double hrate = double(targetheight)/double(height);

	//计算新的器官宽度和高度
	neworganRect.width = wrate * organRect.width;
	neworganRect.height = hrate * organRect.height;

	//计算新的器官的起始x
	int diffx = organRect.x - x1;//原始器官x到脸左侧的距离
	int newdiffx = diffx * wrate;//根据比例算出新器官x到脸左侧的距离
	neworganRect.x = newdiffx + targetx1;


	//计算新的器官的起始y
	int diffy = organRect.y - y1;//原始器官y到脸上侧的距离
	int newdiffy = diffy * hrate;//根据比例算出新器官y到脸上侧的距离
	neworganRect.y = newdiffy + targety1;

	//如上算出了新器官位置的的xywh四个参数
}

//这个函数粘贴器官到人脸
//faceimage 人脸动漫图
//asm_shape 人脸特征点集合
//organRect 器官所在区域(原始人脸上的区域)
//strorganfilename 选出的器官文件名称
void pasteOrgantoFace(IplImage *faceimage,asm_shape shape,CvRect organRect,char *strorganfilename)
{
	//假定人脸轮廓的特征点0和14在高度的0.455处
	int targety1 = faceimage->height * 0.475;//目标y
	int targety2 = faceimage->height;//目标y

	//寻找这个高度处脸的宽度，也就是黑的起止部分宽度
	//这个高度处脸的宽度 ，不一定等于图像的宽度（不一定处在最宽的地方）
	IplImage *bwfaceimage = cvCreateImage(cvGetSize(faceimage),8,1);
	IplImage *grayface = NULL;
	rgb2gray(faceimage,&grayface);//转为灰度
	ThresholdOtsu(grayface,bwfaceimage,CV_THRESH_BINARY);//二值化人脸
	int targetx1,targetx2 = 0;//人脸特征点0 14点的x
	int i;
	for(i=0;i<faceimage->width;i++)
	{
		double value = cvGetReal2D(bwfaceimage,targety1,i);
		if(value <= 0.1)
		{
			targetx1 = i;//如果遇到黑点，就停止
			break;
		}
	}

	for(i=faceimage->width-1;i>=0;i--)
	{
		double value = cvGetReal2D(bwfaceimage,targety1,i);
		if(value <= 0.1)
		{
			targetx2 = i;//如果遇到黑点，就停止
			break;
		}
	}

	CvRect neworganRect;
	computeNewRect(shape,targetx1,targetx2,targety1,targety2,organRect,neworganRect);//算出要粘贴的区域出来
	IplImage *organimage = cvLoadImage(strorganfilename);//载入器官图像
	IplImage *copyorganimage = NULL;
	preproc(organimage,&copyorganimage,cvSize(neworganRect.width,neworganRect.height),PROC_ORI);//将原图处理下，剪切出来，得到指定大小
	//这种情况下，不能直接粘贴，因为图像有白色背景，所以需要设置个mask，只粘贴需要的部分
	IplImage *copyorganimagemask = NULL;
	preproc(copyorganimage,&copyorganimagemask,cvSize(neworganRect.width,neworganRect.height),PROC_BW);//将原图二值化，剪切出来，得到指定大小
	ThresholdOtsu(copyorganimagemask,copyorganimagemask,CV_THRESH_BINARY_INV);	//自动阈值二值化,黑的变白，白的变黑，这样只复制原来黑的地方
	
	//cvNamedWindow("器官");
	//cvShowImage("器官",copyorganimage);
	//cvNamedWindow("器官mask");
	//cvShowImage("器官mask",copyorganimagemask);

	cvSetImageROI(faceimage,neworganRect);//在图中选出区域,设置热点区域
	cvCopy(copyorganimage,faceimage,copyorganimagemask);//复制到faceimage的热点区域，依靠mask的黑白来选择要复制的部分
	cvResetImageROI(faceimage);

	cvReleaseImage(&grayface);
	cvReleaseImage(&bwfaceimage);
	cvReleaseImage(&organimage);
	cvReleaseImage(&copyorganimage);
	cvReleaseImage(&copyorganimagemask);
}


void selectFace(IplImage *image,asm_shape shape,int samplenum,char *strfacefilename)
{
	CvRect r;
	////选择脸型出来
	IplImage *flatimg = cvCreateImage(cvGetSize(image),8,1);//黑板图像
	cvSet(flatimg,CV_RGB(0,0,0));
	CvPoint point1,point2;
	int i=0;
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
	
		cvLine(flatimg,point1,point2,CV_RGB(255,255,255),8);
	}
	//cvNamedWindow("aa",0);
	//cvShowImage("aa",flatimg);
	
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;
	//int comp_count = 1;//增加的
	int Nc = cvFindContours(
		flatimg,
		storage,
		&first_contour,
		sizeof(CvContour),
		CV_RETR_LIST
		);

	

	double minmatchvalue = DBL_MAX;//最小的比较结果
	int matchednum = 1;//对应的匹配上的序号
	for(i=1;i<=samplenum;i++)
	{
		sprintf(strfacefilename,"五官素材\\脸型\\脸型%d.png",i);
		IplImage *imagetmp1 = cvLoadImage(strfacefilename);//读取人脸模型
		

		IplImage *faceimage1 = NULL;
		cropimage(imagetmp1,&faceimage1,CV_THRESH_BINARY);//剪切出紧贴的人脸出来
		IplImage *grayface = NULL;
		rgb2gray(faceimage1,&grayface);//转为灰度
		//ThresholdOtsu(grayface,bwfaceimage,CV_THRESH_BINARY);//二值化人脸

		r.x = 0;
		r.y = faceimage1->height * 0.475;//目标y
		r.width = faceimage1->width;
		r.height = faceimage1->height-r.y;

		IplImage *cropfaceimage = cvCreateImage(cvSize(r.width,r.height),grayface->depth,grayface->nChannels);
		cvSetImageROI(grayface,r);//在图中选出区域,设置热点区域
		cvCopy(grayface,cropfaceimage);//复制faceimage的热点区域到cropfaceimage
		cvResetImageROI(grayface);

		ThresholdOtsu(cropfaceimage,cropfaceimage,CV_THRESH_BINARY_INV);//二值化人脸

		CvSeq* first_contour2 = NULL;
		int Nc = cvFindContours(
		cropfaceimage,
		storage,
		&first_contour2,
		sizeof(CvContour),
		CV_RETR_LIST
		);

		double matchvalue = cvMatchShapes(first_contour,first_contour2,CV_CONTOURS_MATCH_I1);
		printf("matchvalue = %lf\r\n",matchvalue);
		if(matchvalue < minmatchvalue)
		{
			minmatchvalue = matchvalue;
			matchednum = i;
		}

	/*	cvNamedWindow("aa",0);
	cvShowImage("aa",cropfaceimage);*/
	}

	//char strfacefilename[256] = {0};
	sprintf(strfacefilename,"五官素材\\脸型\\脸型%d.png",matchednum);
	printf("%s\r\n",strfacefilename);
}

int main()
{
	asmfitting fit_asm;//特征点检测模型
	char* model_name = "my68-1d.amf";//模型文件
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//人脸定位需要的xml
	char* strfilename = "男1.jpg";//要读取测试的图像文件名
	int samplenum = 6;

	printf("%s\r\n",strfilename);

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

	IplImage* gray = NULL;//灰度图像
	rgb2gray(image,&gray);//转为灰度图像
	IplImage* edge = cvCloneImage(gray);
	cvCanny(gray,edge,50,100);
	cvNamedWindow("边缘",0);
	cvShowImage("边缘",edge);

	//初始化寻找特征点
	InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	fit_asm.Fitting(shape, image, 20);//迭代寻找特征点，使得更准确

	//找完特征点，后截取出主要的器官出来
	int i,j;

	//寻找匹配左眼区域
	vector<CvPoint> lefteyePoints;//左眼点集合
	CvRect lefteyeRect;//左眼rect区域
	IplImage * lefteyeimage = NULL;
	IplImage * lefteyeimageproc = NULL;
	vector<int> lefteyeIndexs;
	for(i=27;i<=31;i++)
	{
		lefteyeIndexs.push_back(i);
	}
	char strnamelefteye[256] = {0};
	//selectOrganImage(image,shape,lefteyeIndexs,cvSize(60,20),PROC_EDGE,"左眼",
	//	&lefteyeimage,&lefteyeimageproc,lefteyeRect,strnamelefteye);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	selectOrganImage2(image,shape,lefteyeIndexs,cvSize(60,20),"左眼",samplenum,
		&lefteyeimage,&lefteyeimageproc,lefteyeRect,strnamelefteye);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	printf("%s\r\n",strnamelefteye);

	//寻找匹配右眼区域
	vector<CvPoint> righteyePoints;//左眼点集合
	CvRect righteyeRect;//左眼rect区域
	IplImage * righteyeimage = NULL;
	IplImage * righteyeimageproc = NULL;
	vector<int> righteyeIndexs;
	for(i=32;i<=36;i++)
	{
		righteyeIndexs.push_back(i);
	}
	char strnamerighteye[256] = {0};
	//selectOrganImage(image,shape,righteyeIndexs,cvSize(60,20),PROC_BW,"右眼",
	//	&righteyeimage,&righteyeimageproc,righteyeRect,strnamerighteye);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	selectOrganImage2(image,shape,righteyeIndexs,cvSize(60,20),"右眼",samplenum,
		&righteyeimage,&righteyeimageproc,righteyeRect,strnamerighteye);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	printf("%s\r\n",strnamerighteye);

	//寻找匹配左眉区域
	vector<CvPoint> leftbrowPoints;//左眉毛点集合
	CvRect leftbrowRect;//左眉毛rect区域
	IplImage * leftbrowimage = NULL;
	IplImage * leftbrowimageproc = NULL;
	vector<int> leftbrowIndexs;
	for(i=21;i<=26;i++)
	{
		leftbrowIndexs.push_back(i);
	}
	char strnameleftbrow[256] = {0};
	//selectOrganImage(image,shape,leftbrowIndexs,cvSize(60,20),PROC_BW,"左眉",
	//	&leftbrowimage,&leftbrowimageproc,leftbrowRect,strnameleftbrow);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	selectOrganImage2(image,shape,leftbrowIndexs,cvSize(60,20),"左眉",samplenum,
		&leftbrowimage,&leftbrowimageproc,leftbrowRect,strnameleftbrow);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	printf("%s\r\n",strnameleftbrow);

	//寻找匹配右眉毛区域
	vector<CvPoint> rightbrowPoints;//右眉点集合
	CvRect rightbrowRect;//右眉rect区域
	IplImage * rightbrowimage = NULL;
	IplImage * rightbrowimageproc = NULL;
	vector<int> rightbrowIndexs;
	for(i=15;i<=20;i++)
	{
		rightbrowIndexs.push_back(i);
	}
	char strnamerightbrow[256] = {0};
	//selectOrganImage(image,shape,rightbrowIndexs,cvSize(60,20),PROC_BW,"右眉",
	//	&rightbrowimage,&rightbrowimageproc,rightbrowRect,strnamerightbrow);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	selectOrganImage2(image,shape,rightbrowIndexs,cvSize(60,20),"右眉",samplenum,
		&rightbrowimage,&rightbrowimageproc,rightbrowRect,strnamerightbrow);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	printf("%s\r\n",strnamerightbrow);


	//寻找匹配嘴型区域
	vector<CvPoint> mouthPoints;//右眉点集合
	CvRect mouthRect;//右眉rect区域
	IplImage * mouthimage = NULL;
	IplImage * mouthimageproc = NULL;
	vector<int> mouthIndexs;
	for(i=48;i<=66;i++)
	{
		mouthIndexs.push_back(i);
	}
	char strnamemouth[256] = {0};
	//selectOrganImage(image,shape,mouthIndexs,cvSize(60,20),PROC_EDGE,"唇型",
	//	&mouthimage,&mouthimageproc,mouthRect,strnamemouth);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	selectOrganImage2(image,shape,mouthIndexs,cvSize(60,20),"唇型",samplenum,
		&mouthimage,&mouthimageproc,mouthRect,strnamemouth);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	//IplImage *selectimage = cvLoadImage(strname);
	printf("%s\r\n",strnamemouth);

	//寻找匹配鼻型区域
	vector<CvPoint> nosePoints;//鼻子点集合
	CvRect noseRect;//鼻子rect区域
	IplImage * noseimage = NULL;
	IplImage * noseimageproc = NULL;
	vector<int> noseIndexs;
	for(i=38;i<=44;i++)
	{
		noseIndexs.push_back(i);
	}
	char strnamenose[256] = {0};
	//selectOrganImage(image,shape,noseIndexs,cvSize(60,50),PROC_EDGE,"鼻型",
	//	&noseimage,&noseimageproc,noseRect,strnamenose);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	selectOrganImage2(image,shape,noseIndexs,cvSize(60,50),"鼻型",samplenum,
		&noseimage,&noseimageproc,noseRect,strnamenose);//这个函数做了一系列处理，选择出了最佳匹配的器官图像
	//IplImage *selectimage = cvLoadImage(strname);
	printf("%s\r\n",strnamenose);


	//选择脸型
	char strfacefilename[256] = {0};
	selectFace(image,shape,6,strfacefilename);//选择脸型
	

	IplImage *imagetmp = cvLoadImage(strfacefilename);//读取选出的人脸脸型
	//IplImage *imagetmp = cvLoadImage("五官素材\\脸型\\脸型6.png");//读取人脸模型，暂时还没选脸型，先用1吧
	IplImage *faceimage = NULL;
	cropimage(imagetmp,&faceimage,CV_THRESH_BINARY);//剪切出紧贴的人脸出来
	//cvNamedWindow("剪切人脸", 0);
	//cvShowImage("剪切人脸", faceimage);	
	

	pasteOrgantoFace(faceimage,shape,lefteyeRect,strnamelefteye);//粘贴左眼
	pasteOrgantoFace(faceimage,shape,righteyeRect,strnamerighteye);//粘贴右眼
	pasteOrgantoFace(faceimage,shape,leftbrowRect,strnameleftbrow);//粘贴左眉毛
	pasteOrgantoFace(faceimage,shape,rightbrowRect,strnamerightbrow);//粘贴右眉毛
	pasteOrgantoFace(faceimage,shape,mouthRect,strnamemouth);//粘贴嘴巴

	//鼻子的有些特殊，训练计算的时候用的是38-44点，37和45点没用
	//粘贴的时候，为了合适，用上37,35点，计算出区域
	CvRect noseRect2;//鼻子rect区
	vector<int> noseIndexs2;
	for(i=37;i<=45;i++)
	{
		noseIndexs2.push_back(i);
	}
	vector<CvPoint> organPoints;//特征点集合
	
	getOrganRect(shape,noseIndexs2, noseRect2,0);//得到原图的鼻子区域RECT
	//这里感觉 用noseRect比noseRect2合适
	pasteOrgantoFace(faceimage,shape,noseRect2,strnamenose);//粘贴鼻子
	

	cvNamedWindow("动漫化", 0);
	cvShowImage("动漫化", faceimage);	

	cvSaveImage("动漫化.jpg", faceimage);	
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

	//将各个器官以及处理的效果图存下来，以便分析
	cvSaveImage("左眼.jpg",lefteyeimage);
	cvSaveImage("左眼处理.jpg", lefteyeimageproc);	
	cvSaveImage("右眼.jpg",righteyeimage);
	cvSaveImage("右眼处理.jpg", righteyeimageproc);	
	cvSaveImage("左眉.jpg",leftbrowimage);
	cvSaveImage("左眉处理.jpg", leftbrowimageproc);	
	cvSaveImage("右眉.jpg",rightbrowimage);
	cvSaveImage("右眉处理.jpg", rightbrowimageproc);	
	cvSaveImage("嘴巴.jpg",mouthimage);
	cvSaveImage("嘴巴处理.jpg", mouthimageproc);	
	cvSaveImage("鼻子.jpg",noseimage);
	cvSaveImage("鼻子处理.jpg", noseimageproc);	

	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}