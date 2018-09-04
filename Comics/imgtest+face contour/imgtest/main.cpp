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
	clock_t start_time=clock();
	asmfitting fit_asm;//��������ģ��
	char* model_name = "my68-1d.amf";//ģ���ļ�
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//������λ��Ҫ��xml
	char* strfilename = "��3.jpg";//Ҫ��ȡ���Ե�ͼ���ļ���


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
	fit_asm.Fitting(shape, image, 10);//����Ѱ�������㣬ʹ�ø�׼ȷ
	shape[9].x = shape[9].x -9;
	shape[10].x = shape[10].x -5;
	clock_t end_time=clock();
    cout<< "������ٶ�λʱ��: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
	//����һ���װ�ͼ��������������ͼ
	IplImage *flatimg = cvCreateImage(cvGetSize(image),8,1);
	cvSet(flatimg,CV_RGB(255,255,255));

	int i,j;
	CvPoint point1;
	CvPoint point2;
	//��������
	for(i=27;i<30;i++)
	{
		//��1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//��2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//���Ʊպϵ����ߣ�β������
	int x = cvRound(shape[27].x);
	int y = cvRound(shape[27].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);

	//��������
	for(i=32;i<35;i++)
	{
		//��1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//��2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//���Ʊպϵ����ߣ�β������
	x = cvRound(shape[32].x);
	y = cvRound(shape[32].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);


	//������üë
	for(i=21;i<26;i++)
	{
		//��1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//��2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//���Ʊպϵ����ߣ�β������
	x = cvRound(shape[21].x);
	y = cvRound(shape[21].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);

	//������üë
	for(i=15;i<20;i++)
	{
		//��1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//��2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//���Ʊպϵ����ߣ�β������
	x = cvRound(shape[15].x);
	y = cvRound(shape[15].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);

	//���Ʊ���
	vector<int> noseIndexs;//�ҳ���Ҫ���Ƶ�˳��
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
	//���Ʊ�������
	for(i=0;i<noseIndexs.size()-1;i++)
	{
		int index = noseIndexs[i];
		//��1
		int x = cvRound(shape[index].x);
		int y = cvRound(shape[index].y);	
		
		point1.x = x;
		point1.y = y;

		index = noseIndexs[i+1];//��һ����
		//��2
		x = cvRound(shape[index].x);
		y = cvRound(shape[index].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}


	//�����������
	for(i=48;i<59;i++)
	{
		//��1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//��2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}
	//���Ʊպϵ����ߣ�β������
	x = cvRound(shape[48].x);
	y = cvRound(shape[48].y);		
	point1.x = x;
	point1.y = y;
	cvLine(flatimg,point2,point1,CV_RGB(0,0,0),2);


		//��������
	for(i=0;i<14;i++)
	{
		//��1
		int x = cvRound(shape[i].x);
		int y = cvRound(shape[i].y);	
		
		point1.x = x;
		point1.y = y;

		//��2
		x = cvRound(shape[i+1].x);
		y = cvRound(shape[i+1].y);
		point2.x = x;
		point2.y = y;
	
		cvLine(flatimg,point1,point2,CV_RGB(0,0,0),2);
	}

	cvNamedWindow("����ͼ",1);
	cvShowImage("����ͼ",flatimg);


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
	cvNamedWindow("������",1);
	cvShowImage("������",image);



	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}