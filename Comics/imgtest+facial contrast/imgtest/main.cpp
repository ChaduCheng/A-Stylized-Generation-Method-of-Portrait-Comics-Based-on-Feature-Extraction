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
	PROC_GRAY,//�Ҷ�
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


//Ԥ����,��Ҫ��תΪ�Ҷȣ���ֵ������С��һ�ȵ�
void preproc(IplImage* inimage,IplImage** outimage,CvSize siz,PROCTYPE type)
{
	
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

//���������ȡ����ͼ������
//image ԭʼͼ��
//indexNumber ���ٶ�Ӧ�ĵ㼯�����
//shape ���������������
//organimage Ҫ�õ�������ͼ��
//organRect �õ�������ͼ��λ��Rect
//points ���ٵ�λ�ü���
void getOrganImage(IplImage *image,asm_shape shape,vector<int> indexNumber, 
	IplImage **organimage,CvRect &organRect,vector<CvPoint> &points)
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
	int diffheight = height * 0.1;
	int diffwidth = width * 0.1;

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


void getPCA(const char* strname,int num,PCA &pcaObject,Mat &alltraindataPCA)
{
	int i;
	char strfilename[256] = {0};

	Mat alltraindata = Mat();

	for(i=1;i<=num;i++)
	{
		sprintf(strfilename,"����ز�\\%s\\%s%d.png",strname,strname,i);
		IplImage * image = cvLoadImage(strfilename);
		IplImage* imageproc = NULL;//Ԥ����ͼ��
		preproc(image,&imageproc,cvSize(60,20),PROC_BW);//ͼ���������֮ǰ��Ԥ������

		
		//Ϊ�˼���pca�������������£�ÿ��ͼ������תΪһ�еľ���
		//Mat img(imageproc);//תΪMat��ʽ
		Mat img = cv::cvarrToMat(imageproc);
		Mat doubleimg;//Ϊ�����ݸ��õĴ���תΪ0-1֮�両����
		img.convertTo(doubleimg,CV_64F,double(1) /double(255),0);
		Mat rowimg = doubleimg.reshape(0,1);//����Ϊһ�еľ���
		alltraindata.push_back(rowimg);//��һ�д洢��picDatas��

		pcaObject(alltraindata,Mat(), CV_PCA_DATA_AS_ROW,30);//�ȵõ�PCA����������pcaObject��
		alltraindataPCA = pcaObject.project(alltraindata);//��ԭʼ����ͨ��PCA����ͶӰ,��ά�������ΪalltraindataPCA

		cvReleaseImage(&imageproc);
	}
}

//����ŷʽ����
double Eucdiatance(Mat feature1,Mat feature2)
{
	double dist = 0;

	Mat diff = feature1 - feature2;//�������
	int rows = diff.rows;
	int cols = diff.cols;

	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			dist = dist + pow(diff.at<double>(i,j),2);//���߶�Ӧά�ȵ�ֵ�����ƽ����
		}
	}
	

	return dist;
}

int main()
{
	asmfitting fit_asm;//��������ģ��
	char* model_name = "my68-1d.amf";//ģ���ļ�
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//������λ��Ҫ��xml
	char* strfilename = "5.jpg";//Ҫ��ȡ���Ե�ͼ���ļ���

	PCA pcaObject;//PCA����
	Mat alltraindataPCA = Mat();//PCA��ά�������
	int samplenum = 6;
	getPCA("����",samplenum,pcaObject,alltraindataPCA);//ѵ��pca

	if(fit_asm.Read(model_name) == false)//װ��������ģ��
	{
		printf("������ģ��װ��ʧ��\r\n");
		return -1;
	}
	
	if(init_detect_cascade(cascade_name) == false)//װ���������xml
	{
		printf("�������ģ��װ��ʧ��\r\n");
		return -1;
	}
	
	IplImage * image = cvLoadImage(strfilename);
	if(image == 0)
	{
		printf("��ͼ��ʧ��: %s\r\n", strfilename);
		return -1;
	}

	asm_shape shape, detshape;
	bool flag =detect_one_face(detshape, image);//��ͼ��Ѱ��������Ѱ������ĵĵ�����

	if(!flag) 
	{	
		printf("û�м�⵽����\r\n");
		return -1;
	}



	//��ʼ��Ѱ��������
	InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	fit_asm.Fitting(shape, image, 20);//����Ѱ�������㣬ʹ�ø�׼ȷ

	//���������㣬���ȡ����Ҫ�����ٳ���
	int i,j;
	//Ѱ����������
	vector<CvPoint> lefteyePoints;//���۵㼯��
	CvRect lefteyeRect;//����rect����
	IplImage * lefteyeimage = NULL;
	vector<int> lefteyeIndexs;
	for(i=27;i<=31;i++)
	{
		lefteyeIndexs.push_back(i);
	}
	getOrganImage(image,shape,lefteyeIndexs, &lefteyeimage,lefteyeRect,lefteyePoints);//Ѱ�����ٳ���


	//Ȼ���ֵ��
	IplImage *bwlefteyeimage = NULL;//����ռ�
	preproc(lefteyeimage,&bwlefteyeimage,cvSize(60,20),PROC_BW);//ͼ���������֮ǰ��Ԥ������

	//Ѱ���زĿ�����ӽ���
	//Mat img(bwlefteyeimage);//תΪMat��ʽ
	Mat img = cv::cvarrToMat(bwlefteyeimage);
	Mat doubleimg;//Ϊ�����ݸ��õĴ���תΪ0-1֮�両����
	img.convertTo(doubleimg,CV_64F,double(1) /double(255),0);
	Mat rowimg = doubleimg.reshape(0,1);//����Ϊһ�еľ���
	
	Mat imgPCA = pcaObject.project(rowimg);//��ԭʼ����ͨ��PCA����ͶӰ,��ά�������ΪimgPCA
			
	int alltrainpicnum = alltraindataPCA.rows;//����ѵ����������
	double minvalue =DBL_MAX;
	int mintype = -1;//������С�ģ�����Ӧ������
	double dist = 0;
	for(int i=0;i<alltrainpicnum;i++)//��������ͼ
	{
		Mat tmp = alltraindataPCA.row(i);//ȡ��һ�����ݣ�Ҳ����һ��ͼ�Ľ�ά�������
		double dist = Eucdiatance(imgPCA,tmp);//����ŷ�Ͼ���
		if(dist < minvalue)//�ҳ���С�ľ��룬�Լ���������
		{
			minvalue = dist;
			mintype = i+1;//�������б���ȡ����Ӧ������
		}
	}

	char strname[256] = {0};
	sprintf(strname,"����ز�\\����\\����%d.png",mintype);
	IplImage *selectimage = cvLoadImage(strname);
	
	printf("%s\r\n",strname);

	//��ʾ������ͼ��
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3, 0, 1, 8);
	int keypointsnum = shape.NPoints();//�ؼ�����
	for (j=0;j<keypointsnum;j++)
	{
		float x = shape[j].x;
		float y = shape[j].y;
				
		CvPoint centerpoint;
		centerpoint.x = cvRound(x);
		centerpoint.y = cvRound(y);
		cvCircle( image, centerpoint ,1 , CV_RGB(255,0,0),1 );		
	}

	cvNamedWindow("������", 0);
	cvShowImage("������", image);	

	cvNamedWindow("����", 1);
	cvShowImage("����", bwlefteyeimage);	

	cvNamedWindow("����ѡ��", 1);
	cvShowImage("����ѡ��", selectimage);	

	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}