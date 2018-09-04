#include "mymethod.h"

#include <stdlib.h>


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

void getPCA2(const char* strname,int num,PCA &pcaObject,Mat &alltraindataPCA,CvSize siz)
{
	int i;
	char strfilename[256] = {0};

	Mat alltraindata = Mat();

	for(i=1;i<=num;i++)
	{
		sprintf(strfilename,"����ز�\\%sѵ��\\%s%d.png",strname,strname,i);
		IplImage * image = cvLoadImage(strfilename);
		IplImage* imageproc = NULL;//Ԥ����ͼ��
		preproc(image,&imageproc,siz,PROC_GRAY);//ͼ���������֮ǰ��Ԥ������

		
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
	IplImage **organimage,IplImage **organimageproc,CvRect &organRect,char *strfilename)
{
	PCA pcaObject;//PCA����
	Mat alltraindataPCA = Mat();//PCA��ά�������
	int samplenum = 6;
	getPCA(strtypename,samplenum,pcaObject,alltraindataPCA,siz);//ѵ��pca

	//������������ţ��ҳ����ٳ���
	int i,j;
	//Ѱ����������
	vector<CvPoint> organPoints;//�����㼯��
	
	getOrganImage(image,shape,indexNumber, organimage,organRect,organPoints);//Ѱ�����ٳ���

	//Ȼ��Ԥ����
	preproc(*organimage,organimageproc,siz,proctype);//ͼ���������֮ǰ��Ԥ������

	//Ѱ���زĿ�����ӽ���
	Mat img(*organimageproc);//תΪMat��ʽ
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

//����������زĿ�ѡȡһ������ʵĳ���
//image ԭʼͼ��
//indexNumber ���ٶ�Ӧ�ĵ㼯�����
//shape ���������������
//siz ������ͳһ��С
//PROCTYPE type �������� BW��ֵ��  EDGE��Ե����
//strtypename ��������
//strfilename ��ѡ����ͼƬ�ļ���
void selectOrganImage2(IplImage *image,asm_shape shape,vector<int> indexNumber,CvSize siz,const char*strtypename,int samplenum,
	IplImage **organimage,IplImage **organimageproc,CvRect &organRect,char *strfilename)
{
	PCA pcaObject;//PCA����
	Mat alltraindataPCA = Mat();//PCA��ά�������
	getPCA2(strtypename,samplenum,pcaObject,alltraindataPCA,siz);//ѵ��pca

	//������������ţ��ҳ����ٳ���
	int i,j;
	//Ѱ����������
	vector<CvPoint> organPoints;//�����㼯��
	
	getOrganImage(image,shape,indexNumber, organimage,organRect,organPoints);//Ѱ�����ٳ���

	//Ȼ��Ԥ����
	preproc(*organimage,organimageproc,siz,PROC_GRAY);//ͼ���������֮ǰ��Ԥ������

	//Ѱ���زĿ�����ӽ���
	Mat img(*organimageproc);//תΪMat��ʽ
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

//�������������µ�Rect����
//targetx1 targetx2 ����������������x����
//targety1 �������ĸߵ�y����
//targety2 �������ĵ͵�y����
//organRect ԭʼ��������
//neworganRect �õ���Ҫճ������������
void computeNewRect(asm_shape shape,int targetx1,int targetx2,int targety1,int targety2,CvRect organRect,CvRect &neworganRect)
{
	//Ҫճ����Ŀ�����ĸ߶ȿ��
	int targetwidth = targetx2 - targetx1 +1;//�������Ŀ��
	int targetheight = targety2 - targety1 + 1;//�������ĸ߶�

	//ԭʼ���Ŀ�Ⱥ͸߶ȼ�����
	int x1 = shape[0].x;
	int x2 = shape[14].x;
	int width = x2-x1+1;
	int y1 = shape[0].y;
	int y2 = shape[7].y;
	int height = y2-y1+1;

	//�������
	double wrate = double(targetwidth)/double(width);
	double hrate = double(targetheight)/double(height);

	//�����µ����ٿ�Ⱥ͸߶�
	neworganRect.width = wrate * organRect.width;
	neworganRect.height = hrate * organRect.height;

	//�����µ����ٵ���ʼx
	int diffx = organRect.x - x1;//ԭʼ����x�������ľ���
	int newdiffx = diffx * wrate;//���ݱ������������x�������ľ���
	neworganRect.x = newdiffx + targetx1;


	//�����µ����ٵ���ʼy
	int diffy = organRect.y - y1;//ԭʼ����y�����ϲ�ľ���
	int newdiffy = diffy * hrate;//���ݱ������������y�����ϲ�ľ���
	neworganRect.y = newdiffy + targety1;

	//���������������λ�õĵ�xywh�ĸ�����
}

//�������ճ�����ٵ�����
//faceimage ��������ͼ
//asm_shape ���������㼯��
//organRect ������������(ԭʼ�����ϵ�����)
//strorganfilename ѡ���������ļ�����
void pasteOrgantoFace(IplImage *faceimage,asm_shape shape,CvRect organRect,char *strorganfilename)
{
	//�ٶ�����������������0��14�ڸ߶ȵ�0.455��
	int targety1 = faceimage->height * 0.475;//Ŀ��y
	int targety2 = faceimage->height;//Ŀ��y

	//Ѱ������߶ȴ����Ŀ�ȣ�Ҳ���Ǻڵ���ֹ���ֿ��
	//����߶ȴ����Ŀ�� ����һ������ͼ��Ŀ�ȣ���һ���������ĵط���
	IplImage *bwfaceimage = cvCreateImage(cvGetSize(faceimage),8,1);
	IplImage *grayface = NULL;
	rgb2gray(faceimage,&grayface);//תΪ�Ҷ�
	ThresholdOtsu(grayface,bwfaceimage,CV_THRESH_BINARY);//��ֵ������
	int targetx1,targetx2 = 0;//����������0 14���x
	int i;
	for(i=0;i<faceimage->width;i++)
	{
		double value = cvGetReal2D(bwfaceimage,targety1,i);
		if(value <= 0.1)
		{
			targetx1 = i;//��������ڵ㣬��ֹͣ
			break;
		}
	}

	for(i=faceimage->width-1;i>=0;i--)
	{
		double value = cvGetReal2D(bwfaceimage,targety1,i);
		if(value <= 0.1)
		{
			targetx2 = i;//��������ڵ㣬��ֹͣ
			break;
		}
	}

	CvRect neworganRect;
	computeNewRect(shape,targetx1,targetx2,targety1,targety2,organRect,neworganRect);//���Ҫճ�����������
	IplImage *organimage = cvLoadImage(strorganfilename);//��������ͼ��
	IplImage *copyorganimage = NULL;
	preproc(organimage,&copyorganimage,cvSize(neworganRect.width,neworganRect.height),PROC_ORI);//��ԭͼ�����£����г������õ�ָ����С
	//��������£�����ֱ��ճ������Ϊͼ���а�ɫ������������Ҫ���ø�mask��ֻճ����Ҫ�Ĳ���
	IplImage *copyorganimagemask = NULL;
	preproc(copyorganimage,&copyorganimagemask,cvSize(neworganRect.width,neworganRect.height),PROC_BW);//��ԭͼ��ֵ�������г������õ�ָ����С
	ThresholdOtsu(copyorganimagemask,copyorganimagemask,CV_THRESH_BINARY_INV);	//�Զ���ֵ��ֵ��,�ڵı�ף��׵ı�ڣ�����ֻ����ԭ���ڵĵط�
	
	//cvNamedWindow("����");
	//cvShowImage("����",copyorganimage);
	//cvNamedWindow("����mask");
	//cvShowImage("����mask",copyorganimagemask);

	cvSetImageROI(faceimage,neworganRect);//��ͼ��ѡ������,�����ȵ�����
	cvCopy(copyorganimage,faceimage,copyorganimagemask);//���Ƶ�faceimage���ȵ���������mask�ĺڰ���ѡ��Ҫ���ƵĲ���
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
	////ѡ�����ͳ���
	IplImage *flatimg = cvCreateImage(cvGetSize(image),8,1);//�ڰ�ͼ��
	cvSet(flatimg,CV_RGB(0,0,0));
	CvPoint point1,point2;
	int i=0;
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
	
		cvLine(flatimg,point1,point2,CV_RGB(255,255,255),8);
	}
	//cvNamedWindow("aa",0);
	//cvShowImage("aa",flatimg);
	
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;
	//int comp_count = 1;//���ӵ�
	int Nc = cvFindContours(
		flatimg,
		storage,
		&first_contour,
		sizeof(CvContour),
		CV_RETR_LIST
		);

	

	double minmatchvalue = DBL_MAX;//��С�ıȽϽ��
	int matchednum = 1;//��Ӧ��ƥ���ϵ����
	for(i=1;i<=samplenum;i++)
	{
		sprintf(strfacefilename,"����ز�\\����\\����%d.png",i);
		IplImage *imagetmp1 = cvLoadImage(strfacefilename);//��ȡ����ģ��
		

		IplImage *faceimage1 = NULL;
		cropimage(imagetmp1,&faceimage1,CV_THRESH_BINARY);//���г���������������
		IplImage *grayface = NULL;
		rgb2gray(faceimage1,&grayface);//תΪ�Ҷ�
		//ThresholdOtsu(grayface,bwfaceimage,CV_THRESH_BINARY);//��ֵ������

		r.x = 0;
		r.y = faceimage1->height * 0.475;//Ŀ��y
		r.width = faceimage1->width;
		r.height = faceimage1->height-r.y;

		IplImage *cropfaceimage = cvCreateImage(cvSize(r.width,r.height),grayface->depth,grayface->nChannels);
		cvSetImageROI(grayface,r);//��ͼ��ѡ������,�����ȵ�����
		cvCopy(grayface,cropfaceimage);//����faceimage���ȵ�����cropfaceimage
		cvResetImageROI(grayface);

		ThresholdOtsu(cropfaceimage,cropfaceimage,CV_THRESH_BINARY_INV);//��ֵ������

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
	sprintf(strfacefilename,"����ز�\\����\\����%d.png",matchednum);
	printf("%s\r\n",strfacefilename);
}

int main()
{
	asmfitting fit_asm;//��������ģ��
	char* model_name = "my68-1d.amf";//ģ���ļ�
	char* cascade_name = "haarcascade_frontalface_alt2.xml";//������λ��Ҫ��xml
	char* strfilename = "��1.jpg";//Ҫ��ȡ���Ե�ͼ���ļ���
	int samplenum = 6;

	printf("%s\r\n",strfilename);

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

	IplImage* gray = NULL;//�Ҷ�ͼ��
	rgb2gray(image,&gray);//תΪ�Ҷ�ͼ��
	IplImage* edge = cvCloneImage(gray);
	cvCanny(gray,edge,50,100);
	cvNamedWindow("��Ե",0);
	cvShowImage("��Ե",edge);

	//��ʼ��Ѱ��������
	InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	fit_asm.Fitting(shape, image, 20);//����Ѱ�������㣬ʹ�ø�׼ȷ

	//���������㣬���ȡ����Ҫ�����ٳ���
	int i,j;

	//Ѱ��ƥ����������
	vector<CvPoint> lefteyePoints;//���۵㼯��
	CvRect lefteyeRect;//����rect����
	IplImage * lefteyeimage = NULL;
	IplImage * lefteyeimageproc = NULL;
	vector<int> lefteyeIndexs;
	for(i=27;i<=31;i++)
	{
		lefteyeIndexs.push_back(i);
	}
	char strnamelefteye[256] = {0};
	//selectOrganImage(image,shape,lefteyeIndexs,cvSize(60,20),PROC_EDGE,"����",
	//	&lefteyeimage,&lefteyeimageproc,lefteyeRect,strnamelefteye);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	selectOrganImage2(image,shape,lefteyeIndexs,cvSize(60,20),"����",samplenum,
		&lefteyeimage,&lefteyeimageproc,lefteyeRect,strnamelefteye);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	printf("%s\r\n",strnamelefteye);

	//Ѱ��ƥ����������
	vector<CvPoint> righteyePoints;//���۵㼯��
	CvRect righteyeRect;//����rect����
	IplImage * righteyeimage = NULL;
	IplImage * righteyeimageproc = NULL;
	vector<int> righteyeIndexs;
	for(i=32;i<=36;i++)
	{
		righteyeIndexs.push_back(i);
	}
	char strnamerighteye[256] = {0};
	//selectOrganImage(image,shape,righteyeIndexs,cvSize(60,20),PROC_BW,"����",
	//	&righteyeimage,&righteyeimageproc,righteyeRect,strnamerighteye);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	selectOrganImage2(image,shape,righteyeIndexs,cvSize(60,20),"����",samplenum,
		&righteyeimage,&righteyeimageproc,righteyeRect,strnamerighteye);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	printf("%s\r\n",strnamerighteye);

	//Ѱ��ƥ����ü����
	vector<CvPoint> leftbrowPoints;//��üë�㼯��
	CvRect leftbrowRect;//��üërect����
	IplImage * leftbrowimage = NULL;
	IplImage * leftbrowimageproc = NULL;
	vector<int> leftbrowIndexs;
	for(i=21;i<=26;i++)
	{
		leftbrowIndexs.push_back(i);
	}
	char strnameleftbrow[256] = {0};
	//selectOrganImage(image,shape,leftbrowIndexs,cvSize(60,20),PROC_BW,"��ü",
	//	&leftbrowimage,&leftbrowimageproc,leftbrowRect,strnameleftbrow);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	selectOrganImage2(image,shape,leftbrowIndexs,cvSize(60,20),"��ü",samplenum,
		&leftbrowimage,&leftbrowimageproc,leftbrowRect,strnameleftbrow);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	printf("%s\r\n",strnameleftbrow);

	//Ѱ��ƥ����üë����
	vector<CvPoint> rightbrowPoints;//��ü�㼯��
	CvRect rightbrowRect;//��ürect����
	IplImage * rightbrowimage = NULL;
	IplImage * rightbrowimageproc = NULL;
	vector<int> rightbrowIndexs;
	for(i=15;i<=20;i++)
	{
		rightbrowIndexs.push_back(i);
	}
	char strnamerightbrow[256] = {0};
	//selectOrganImage(image,shape,rightbrowIndexs,cvSize(60,20),PROC_BW,"��ü",
	//	&rightbrowimage,&rightbrowimageproc,rightbrowRect,strnamerightbrow);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	selectOrganImage2(image,shape,rightbrowIndexs,cvSize(60,20),"��ü",samplenum,
		&rightbrowimage,&rightbrowimageproc,rightbrowRect,strnamerightbrow);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	printf("%s\r\n",strnamerightbrow);


	//Ѱ��ƥ����������
	vector<CvPoint> mouthPoints;//��ü�㼯��
	CvRect mouthRect;//��ürect����
	IplImage * mouthimage = NULL;
	IplImage * mouthimageproc = NULL;
	vector<int> mouthIndexs;
	for(i=48;i<=66;i++)
	{
		mouthIndexs.push_back(i);
	}
	char strnamemouth[256] = {0};
	//selectOrganImage(image,shape,mouthIndexs,cvSize(60,20),PROC_EDGE,"����",
	//	&mouthimage,&mouthimageproc,mouthRect,strnamemouth);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	selectOrganImage2(image,shape,mouthIndexs,cvSize(60,20),"����",samplenum,
		&mouthimage,&mouthimageproc,mouthRect,strnamemouth);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	//IplImage *selectimage = cvLoadImage(strname);
	printf("%s\r\n",strnamemouth);

	//Ѱ��ƥ���������
	vector<CvPoint> nosePoints;//���ӵ㼯��
	CvRect noseRect;//����rect����
	IplImage * noseimage = NULL;
	IplImage * noseimageproc = NULL;
	vector<int> noseIndexs;
	for(i=38;i<=44;i++)
	{
		noseIndexs.push_back(i);
	}
	char strnamenose[256] = {0};
	//selectOrganImage(image,shape,noseIndexs,cvSize(60,50),PROC_EDGE,"����",
	//	&noseimage,&noseimageproc,noseRect,strnamenose);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	selectOrganImage2(image,shape,noseIndexs,cvSize(60,50),"����",samplenum,
		&noseimage,&noseimageproc,noseRect,strnamenose);//�����������һϵ�д���ѡ��������ƥ�������ͼ��
	//IplImage *selectimage = cvLoadImage(strname);
	printf("%s\r\n",strnamenose);


	//ѡ������
	char strfacefilename[256] = {0};
	selectFace(image,shape,6,strfacefilename);//ѡ������
	

	IplImage *imagetmp = cvLoadImage(strfacefilename);//��ȡѡ������������
	//IplImage *imagetmp = cvLoadImage("����ز�\\����\\����6.png");//��ȡ����ģ�ͣ���ʱ��ûѡ���ͣ�����1��
	IplImage *faceimage = NULL;
	cropimage(imagetmp,&faceimage,CV_THRESH_BINARY);//���г���������������
	//cvNamedWindow("��������", 0);
	//cvShowImage("��������", faceimage);	
	

	pasteOrgantoFace(faceimage,shape,lefteyeRect,strnamelefteye);//ճ������
	pasteOrgantoFace(faceimage,shape,righteyeRect,strnamerighteye);//ճ������
	pasteOrgantoFace(faceimage,shape,leftbrowRect,strnameleftbrow);//ճ����üë
	pasteOrgantoFace(faceimage,shape,rightbrowRect,strnamerightbrow);//ճ����üë
	pasteOrgantoFace(faceimage,shape,mouthRect,strnamemouth);//ճ�����

	//���ӵ���Щ���⣬ѵ�������ʱ���õ���38-44�㣬37��45��û��
	//ճ����ʱ��Ϊ�˺��ʣ�����37,35�㣬���������
	CvRect noseRect2;//����rect��
	vector<int> noseIndexs2;
	for(i=37;i<=45;i++)
	{
		noseIndexs2.push_back(i);
	}
	vector<CvPoint> organPoints;//�����㼯��
	
	getOrganRect(shape,noseIndexs2, noseRect2,0);//�õ�ԭͼ�ı�������RECT
	//����о� ��noseRect��noseRect2����
	pasteOrgantoFace(faceimage,shape,noseRect2,strnamenose);//ճ������
	

	cvNamedWindow("������", 0);
	cvShowImage("������", faceimage);	

	cvSaveImage("������.jpg", faceimage);	
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

	//�����������Լ������Ч��ͼ���������Ա����
	cvSaveImage("����.jpg",lefteyeimage);
	cvSaveImage("���۴���.jpg", lefteyeimageproc);	
	cvSaveImage("����.jpg",righteyeimage);
	cvSaveImage("���۴���.jpg", righteyeimageproc);	
	cvSaveImage("��ü.jpg",leftbrowimage);
	cvSaveImage("��ü����.jpg", leftbrowimageproc);	
	cvSaveImage("��ü.jpg",rightbrowimage);
	cvSaveImage("��ü����.jpg", rightbrowimageproc);	
	cvSaveImage("���.jpg",mouthimage);
	cvSaveImage("��ʹ���.jpg", mouthimageproc);	
	cvSaveImage("����.jpg",noseimage);
	cvSaveImage("���Ӵ���.jpg", noseimageproc);	

	cvWaitKey(0);			
	cvReleaseImage(&image);

	return 0;
}