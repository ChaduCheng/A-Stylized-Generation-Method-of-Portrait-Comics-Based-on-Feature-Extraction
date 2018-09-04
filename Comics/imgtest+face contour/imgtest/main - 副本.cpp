#include "mymethod.h"




void getPCA(const char* strname,int num,PCA &pcaObject,Mat &alltraindataPCA,CvSize siz)
{
	int i;
	char strfilename[256] = {0};

	Mat alltraindata = Mat();

	for(i=1;i<=num;i++)
	{
		sprintf(strfilename,"����ز�\\%s\\%s%d.png",strname,strname,i);
		IplImage * image = cvLoadImage(strfilename);
		IplImage* imageproc = NULL;//Ԥ����ͼ��
		preproc(image,&imageproc,siz,PROC_BW);//ͼ���������֮ǰ��Ԥ������

		
		//Ϊ�˼���pca�������������£�ÿ��ͼ������תΪһ�еľ���
		Mat img(imageproc);//תΪMat��ʽ
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

//����������زĿ�ѡȡһ������ʵĳ���
//image ԭʼͼ��
//indexNumber ���ٶ�Ӧ�ĵ㼯�����
//shape ���������������
//siz ������ͳһ��С
//PROCTYPE type �������� BW��ֵ��  EDGE��Ե����
//strtypename ��������
//strfilename ��ѡ����ͼƬ�ļ���
void selectOrganImage(IplImage *image,asm_shape shape,vector<int> indexNumber,CvSize siz,PROCTYPE proctype,const char*strtypename,
	IplImage **organimage,CvRect &organRect,char *strfilename)
{
	PCA pcaObject;//PCA����
	Mat alltraindataPCA = Mat();//PCA��ά�������
	int samplenum = 6;
	getPCA(strtypename,samplenum,pcaObject,alltraindataPCA,siz);//ѵ��pca

	//������������ţ��ҳ����ٳ���
	int i,j;
	//Ѱ����������
	vector<Point> organPoints;//�����㼯��
	
	getOrganImage(image,shape,indexNumber, organimage,organRect,organPoints);//Ѱ�����ٳ���

	//Ȼ��Ԥ����
	IplImage *bworganimage = NULL;//����ռ�
	preproc(*organimage,&bworganimage,siz,proctype);//ͼ���������֮ǰ��Ԥ������

	//Ѱ���زĿ�����ӽ���
	Mat img(bworganimage);//תΪMat��ʽ
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

	sprintf(strfilename,"����ز�\\%s\\%s%d.png",strtypename,strtypename,mintype);

}

int main()
{
	asmfitting fit_asm;//��������ģ��
	char* model_name = "my68-1d.amf";//ģ���ļ�
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//������λ��Ҫ��xml
	char* strfilename = "��1.jpg";//Ҫ��ȡ���Ե�ͼ���ļ���


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

	vector<vector<Point>> contours;

	//��ʼ��Ѱ��������
	InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	fit_asm.Fitting(shape, image, 20);//����Ѱ�������㣬ʹ�ø�׼ȷ

	//���������㣬���ȡ����Ҫ�����ٳ���
	int i,j;

	//Ѱ��ƥ����������
	vector<Point> lefteyePoints;//���۵㼯��
	CvRect lefteyeRect;//����rect����
	IplImage * lefteyeimage = NULL;
	vector<int> lefteyeIndexs;
	for(i=27;i<=30;i++)
	{
		lefteyeIndexs.push_back(i);
	
	}
	getOrganImage(image,shape,lefteyeIndexs,&lefteyeimage,lefteyeRect,lefteyePoints);
	contours.push_back(lefteyePoints);

	//Ѱ��ƥ����������
	vector<Point> righteyePoints;//���۵㼯��
	CvRect righteyeRect;//����rect����
	IplImage * righteyeimage = NULL;
	vector<int> righteyeIndexs;
	for(i=32;i<=35;i++)
	{
		righteyeIndexs.push_back(i);
	}
	getOrganImage(image,shape,righteyeIndexs,&righteyeimage,righteyeRect,righteyePoints);
	contours.push_back(righteyePoints);

	//Ѱ��ƥ����ü����
	vector<Point> leftbrowPoints;//��üë�㼯��
	CvRect leftbrowRect;//��üërect����
	IplImage * leftbrowimage = NULL;
	vector<int> leftbrowIndexs;
	for(i=21;i<=26;i++)
	{
		leftbrowIndexs.push_back(i);
	}
	getOrganImage(image,shape,leftbrowIndexs,&leftbrowimage,leftbrowRect,leftbrowPoints);
	contours.push_back(leftbrowPoints);


	//Ѱ��ƥ����üë����
	vector<Point> rightbrowPoints;//��ü�㼯��
	CvRect rightbrowRect;//��ürect����
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
	//IplImage *bwimage = cvCreateImage(cvGetSize(rightbrowimage), 8, 1);//����ռ�
	//	ThresholdOtsu(grayimg,bwimage,CV_THRESH_BINARY);	//�Զ���ֵ��ֵ��


	//Ѱ��ƥ����������
	vector<Point> mouthPoints;//��ü�㼯��
	CvRect mouthRect;//��ürect����
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

	//Ѱ��ƥ���������
	vector<Point> nosePoints;//���ӵ㼯��
	CvRect noseRect;//����rect����
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

	vector<Point> nosePoints2;//���ӵ㼯��
	CvRect noseRect2;//����rect����
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

	vector<Point> facePoints;//���͵㼯��
	CvRect faceRect;//����rect����
	IplImage * faceimage = NULL;
	vector<int> faceIndexs;
	for(i=0;i<=14;i++)
	{
		faceIndexs.push_back(i);
	}
	getOrganImage(image,shape,faceIndexs,&faceimage,faceRect,facePoints);
	contours.push_back(facePoints);

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
		cvCircle( image, centerpoint ,2 , CV_RGB(255,0,0),-1 );		
	}

	Mat flatimg(image->height,image->width,CV_8UC1,Scalar::all(255));//����һ��ȫ��ɫ��ͼ��
	drawContours(flatimg, contours, 0, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 1, CV_RGB(0,0,0),2, 8);
	drawContours(flatimg, contours, 2, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 3, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 4, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 5, CV_RGB(0,0,0), 2, 8);
	drawContours(flatimg, contours, 6, CV_RGB(0,0,0), 2, 8);

	cvNamedWindow("������", 0);
	cvShowImage("������", image);	

	cvNamedWindow("����", 1);
	cvShowImage("����", lefteyeimage);	

	cvNamedWindow("����", 1);
	cvShowImage("����", righteyeimage);	

	cvNamedWindow("��ü", 1);
	cvShowImage("��ü", leftbrowimage);	

	cvNamedWindow("��ü", 1);
	cvShowImage("��ü", rightbrowimage);	

	
	cvNamedWindow("����", 1);
	cvShowImage("����", noseimage);	

	cvNamedWindow("���", 1);
	cvShowImage("���", mouthimage);	

	cvNamedWindow("����", 0);
	imshow("����", flatimg);	

	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}