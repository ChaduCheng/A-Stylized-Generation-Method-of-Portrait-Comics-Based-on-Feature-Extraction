
// beatutyfaceDlg.h : ͷ�ļ�
//

#pragma once

#include "opencv2/opencv.hpp"
#include <vector>
#include "afxwin.h"
#include "CvvImage.h"

using namespace std;
using namespace cv;

// CbeatutyfaceDlg �Ի���
class CbeatutyfaceDlg : public CDialogEx
{
// ����
public:
	CbeatutyfaceDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_BEATUTYFACE_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


	void mycvDrawImage(IplImage *psrc, CRect oRect, HWND Hwnd, CDC * pDC, CRect& oCurImageDrawRect, int nDisplayType);

// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();

	CString m_strImagePath;
	Mat m_loadImg;
	Mat m_processImg;
	Mat m_beautyImg;

	CRect m_oOriginalRect;
	CRect m_oBeautyRect;


	CStatic m_ctlOriginalRect;
	CStatic m_ctlBeautyRect;
	afx_msg void OnBnClickedButton2();
	CString m_strResult;
	afx_msg void OnBnClickedButton3();
};

