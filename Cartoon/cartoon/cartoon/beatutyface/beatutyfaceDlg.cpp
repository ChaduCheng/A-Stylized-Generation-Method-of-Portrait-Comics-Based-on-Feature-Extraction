
// beatutyfaceDlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "beatutyface.h"
#include "beatutyfaceDlg.h"
#include "afxdialogex.h"
#include <vector>
#include "io.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif



using namespace std;
using namespace cv;

//���һ���ļ����ڵ�����ͼ������
void getFiles(string path, vector<string>& files)
{
	//�ļ����
	long   hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮
			//�������,�����б�
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}



// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CbeatutyfaceDlg �Ի���



CbeatutyfaceDlg::CbeatutyfaceDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CbeatutyfaceDlg::IDD, pParent)
	, m_strResult(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CbeatutyfaceDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_OriginalRect, m_ctlOriginalRect);
	DDX_Control(pDX, IDC_BeautyRect, m_ctlBeautyRect);
	DDX_Text(pDX, IDC_Result, m_strResult);
}

BEGIN_MESSAGE_MAP(CbeatutyfaceDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CbeatutyfaceDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CbeatutyfaceDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CbeatutyfaceDlg::OnBnClickedButton3)
END_MESSAGE_MAP()


// CbeatutyfaceDlg ��Ϣ�������

BOOL CbeatutyfaceDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO:  �ڴ���Ӷ���ĳ�ʼ������

	m_ctlOriginalRect.GetWindowRect(&m_oOriginalRect);
	ScreenToClient(m_oOriginalRect);

	m_ctlBeautyRect.GetWindowRect(&m_oBeautyRect);
	ScreenToClient(m_oBeautyRect);

	m_loadImg = Mat::zeros(cvSize(0, 0), CV_8UC3);
	m_beautyImg = Mat::zeros(cvSize(0, 0), CV_8UC3);

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void CbeatutyfaceDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CbeatutyfaceDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}

	//CRect otemp;
	//if (m_loadImg.cols > 0 && m_loadImg.rows > 0)
	//{
	//	mycvDrawImage(&(IplImage)m_loadImg, m_oOriginalRect, GetSafeHwnd(), GetDC(), otemp, 1);
	//}

	//if (m_beautyImg.cols > 0 && m_beautyImg.rows > 0)
	//{
	//	mycvDrawImage(&(IplImage)m_beautyImg, m_oBeautyRect, GetSafeHwnd(), GetDC(), otemp, 1);
	//}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CbeatutyfaceDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CbeatutyfaceDlg::mycvDrawImage(IplImage *psrc, CRect oRect, HWND Hwnd, CDC * pDC, CRect& oCurImageDrawRect, int nDisplayType)
{

	if (!psrc || !pDC || !Hwnd)
	{
		return;
	}
	if (nDisplayType < 0 || nDisplayType>1)
	{
		return;
	}

	// output image size(or oImageRect size)
	int nWidth = 0;
	int nHeight = 0;

	// create a memory DC , and link it with pDC
	CDC CurShowMemDC;
	CurShowMemDC.CreateCompatibleDC(pDC);

	CBitmap CurShowBmp;
	CurShowBmp.CreateCompatibleBitmap(pDC, oRect.Width(), oRect.Height());
	CBitmap* pOldBitmap1 = CurShowMemDC.SelectObject(&CurShowBmp);

	CRect oImageRect;
	float fZoomRatio = 0.0;
	float fSizeRatio = 0.0;

	if (nDisplayType == 1)
	{
		// first draw background as black
		CurShowMemDC.PatBlt(oRect.left, oRect.top, oRect.Width(), oRect.Height(), BLACKNESS);

		fSizeRatio = (float)psrc->width / (float)psrc->height;
		fZoomRatio = (float)psrc->width / (float)oRect.Width();
		if (psrc->height / fZoomRatio > oRect.Height()) // height fit, we should adjust width
		{
			fZoomRatio = (float)psrc->height / (float)oRect.Height();
			oImageRect.left = oRect.left;
			oImageRect.top = oRect.top;
			oImageRect.bottom = oRect.bottom;
			oImageRect.right = oRect.Height()*fSizeRatio + oRect.left;
		}
		else               // width fit, we should adjust height
		{

			oImageRect.left = oRect.left;
			oImageRect.top = oRect.top;
			oImageRect.right = oRect.right;
			oImageRect.bottom = oRect.Width() / fSizeRatio + oRect.top;
		}
		// reset oImageRect's position in order it in the center of oRect
		int nCenterX = oRect.Width() / 2 + oRect.left + 0.5;
		int nCenterY = oRect.Height() / 2 + oRect.top + 0.5;
		nWidth = oImageRect.Width();
		nHeight = oImageRect.Height();
		oImageRect.left = nCenterX - nWidth / 2;
		oImageRect.top = nCenterY - nHeight / 2;

		oImageRect.right = oImageRect.left + nWidth;
		oImageRect.bottom = oImageRect.top + nHeight;


	}
	else if (nDisplayType == 0)
	{
		fZoomRatio = 1.0;
		oImageRect.left = oRect.left;
		oImageRect.top = oRect.top;
		oImageRect.right = oRect.right;
		oImageRect.bottom = oRect.bottom;
	}
	oCurImageDrawRect = oImageRect;
	IplImage *pShowImage = cvCreateImage(cvSize(oImageRect.Width(), oImageRect.Height()), psrc->depth, psrc->nChannels);
	cvResize(psrc, pShowImage, CV_INTER_CUBIC);

	CvvImage Cmyimg;
	HDC hDC;
	hDC = pDC->GetSafeHdc();

	//Cmyimg.CopyOf(pShowImage,1);

	//Cmyimg.DrawToHDC(hDC,&oImageRect);

	Cmyimg.CopyOf(pShowImage, 1);

	//convert oImageRect coordinate from pDC to memory DC

	nWidth = oImageRect.Width();
	nHeight = oImageRect.Height();

	oImageRect.left = oImageRect.left - oRect.left;
	oImageRect.top = oImageRect.top - oRect.top;

	oImageRect.right = oImageRect.left + nWidth;
	oImageRect.bottom = oImageRect.top + nHeight;


	// convert end

	Cmyimg.DrawToHDC(CurShowMemDC.GetSafeHdc(), &oImageRect);
	pDC->BitBlt(oRect.left, oRect.top, oRect.Width(), oRect.Height(), &CurShowMemDC, 0, 0, SRCCOPY);

	CurShowMemDC.DeleteDC();
	::ReleaseDC(Hwnd, hDC);
	cvReleaseImage(&pShowImage);
	return;
}


void CbeatutyfaceDlg::OnBnClickedButton1()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������


	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, "ImageFile(*.*)|*.*||", NULL);
	if (dlg.DoModal() == IDOK)
	{
		m_strImagePath = dlg.GetPathName();
		m_loadImg = imread(m_strImagePath.GetBuffer());
		
	}

	CRect otemp;
	if (m_loadImg.cols>0&&m_loadImg.rows>0)
	{
		mycvDrawImage(&(IplImage)m_loadImg, m_oOriginalRect, GetSafeHwnd(), GetDC(), otemp, 1);
	}
	

	
	m_strResult = "";
	UpdateData(FALSE);
}



int repetitions = 7; // Repetitions for strong cartoon effect.
const int MEDIAN_BLUR_FILTER_SIZE = 7;
const int LAPLACIAN_FILTER_SIZE = 5;
const int EDGES_THRESHOLD = 80;



void CbeatutyfaceDlg::OnBnClickedButton2()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	Mat _labImg;
	cvtColor(m_loadImg, _labImg, CV_RGB2Lab);

	vector<Mat> _labElements;
	split(_labImg, _labElements);

	imwrite("..\\result\\step1_cartoon_labImg.jpg", _labImg);
	imwrite("..\\result\\step1_cartoon_labImg_L.jpg", _labElements[0]);
	imwrite("..\\result\\step1_cartoon_labImg_A.jpg", _labElements[1]);
	imwrite("..\\result\\step1_cartoon_labImg_B.jpg", _labElements[2]);

	for (int i = 0; i < repetitions; i++)
	{
		int ksize = 9; // Filter size. Has a large effect on speed.
		double sigmaColor = 9; // Filter color strength.
		double sigmaSpace = 7; // Spatial strength. Affects speed.
		Mat tmp;
		bilateralFilter(_labElements[0], tmp, ksize, sigmaColor, sigmaSpace);
		bilateralFilter(tmp, _labElements[0], ksize, sigmaColor, sigmaSpace);

		bilateralFilter(_labElements[1], tmp, ksize, sigmaColor, sigmaSpace);
		bilateralFilter(tmp, _labElements[1], ksize, sigmaColor, sigmaSpace);

		bilateralFilter(_labElements[2], tmp, ksize, sigmaColor, sigmaSpace);
		bilateralFilter(tmp, _labElements[2], ksize, sigmaColor, sigmaSpace);
	}

	Mat res_AfterBilaFilter;
	merge(_labElements, res_AfterBilaFilter);
	imwrite("..\\result\\step1_cartoon_labImg_L_ABF.jpg", _labElements[0]);
	cvtColor(res_AfterBilaFilter, res_AfterBilaFilter, CV_Lab2RGB);
	imwrite("..\\result\\step2_cartoon_bilateral.jpg",res_AfterBilaFilter);


	

	Mat  edges;
	Laplacian(_labElements[0], edges, CV_8U, LAPLACIAN_FILTER_SIZE);


	Mat masks;
	threshold(edges, masks, EDGES_THRESHOLD, 255, THRESH_BINARY_INV);

	Mat dst = Mat(cvSize(_labElements[0].cols, _labElements[0].rows), CV_8UC3);
	dst.setTo(0);
	int nLuminance_detaQ = 24;

	for (int y = 0; y < _labElements[0].rows; y++)
	{
		for (int x = 0; x < _labElements[0].cols; x++)
		{
			_labElements[0].at<unsigned char>(y, x) = _labElements[0].at<unsigned char>(y, x) / nLuminance_detaQ*nLuminance_detaQ;
		}
	}
	_labElements[0].copyTo(dst, masks);


	imwrite("..\\result\\step3.1_cartoon_luminanceQuantization.jpg", _labElements[0]);

	Mat img_G0;
	Mat img_G1;
	
	GaussianBlur(_labElements[0], img_G0, Size(1, 1), 0);
	GaussianBlur(img_G0, img_G1, Size(9, 9), 0);
	Mat img_DoG = img_G0 - img_G1;
	Mat img_dog2 = 255 - img_DoG;
	normalize(img_DoG, img_DoG, 255, 0, CV_MINMAX);
	imwrite("..\\result\\step3.2_cartoon_dog.jpg", img_dog2);


	_labElements[0] = 0.1*_labElements[0] + 0.1*img_DoG+0.9*dst;
	

	merge(_labElements, m_beautyImg);
	imwrite("..\\result\\step4_cartoon_combined.jpg", m_beautyImg);
	cvtColor(m_beautyImg, m_beautyImg, CV_Lab2RGB);

	imwrite("..\\result\\step5_cartoon_combine.jpg", m_beautyImg);


	CRect otemp;
	if (m_beautyImg.cols > 0 && m_beautyImg.rows > 0)
	{
		mycvDrawImage(&(IplImage)m_beautyImg, m_oOriginalRect, GetSafeHwnd(), GetDC(), otemp, 1);
	}

	UpdateData(FALSE);

}





void CbeatutyfaceDlg::OnBnClickedButton3()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	Mat grayImage;

	//�ҶȻ�
	cvtColor(m_loadImg, grayImage, CV_BGR2GRAY);
	imwrite("..\\result\\step1_sketch_gray.jpg",grayImage);

	// ������ֵ�˲�������
	medianBlur(grayImage, grayImage, 7);
	imwrite("..\\result\\step2_sketch_midian.jpg", grayImage);
	// Laplacian��Ե���
	Mat edge;
	Laplacian(grayImage, edge, CV_8U, 5);
	imwrite("..\\result\\step3_skect_dog.jpg", edge);
	

	// �Ա�Ե��������ж�ֵ��
	Mat dstImage;
	threshold(edge, dstImage, 127, 255, THRESH_BINARY_INV);// >127 ? 0:255,�ú�ɫ�������
	imwrite("..\\result\\step3_skect_final.jpg", dstImage);

	m_beautyImg = dstImage;

	CRect otemp;
	if (m_beautyImg.cols > 0 && m_beautyImg.rows > 0)
	{
		mycvDrawImage(&(IplImage)m_beautyImg, m_oOriginalRect, GetSafeHwnd(), GetDC(), otemp, 1);
	}

	UpdateData(FALSE);

}
