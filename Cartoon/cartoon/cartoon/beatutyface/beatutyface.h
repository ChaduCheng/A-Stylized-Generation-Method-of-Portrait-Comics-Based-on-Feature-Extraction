
// beatutyface.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CbeatutyfaceApp: 
// �йش����ʵ�֣������ beatutyface.cpp
//

class CbeatutyfaceApp : public CWinApp
{
public:
	CbeatutyfaceApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CbeatutyfaceApp theApp;