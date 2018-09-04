#include "stdio.h"
#include "stdlib.h"

#ifndef _SOME_FUNCTIONS_CPP_123_
#define _SOME_FUNCTIONS_CPP_123_


#define SafeFreeArray(X) {if(X!=NULL) {free(X); X =NULL;}}
#define SafeDeleteArray(X) {if(X!=NULL) {delete[] (X); X =NULL;}}
#define SafeDeleteClass(X) {if(X!=NULL) {delete (X); X =NULL;}}

unsigned char *FUNCTION_ReadBmpFile2(const char *filename, int * ht_t, int * wd_t);
void FUNCTION_YUV420SP_to_BGR24(unsigned char *pYUV420, int wd, int ht, unsigned char* pBGR24, int mode);
void FUNCTION_BGR24_to_YUV420SP(unsigned char *pBGR24, int wd, int ht, unsigned char* pYUV420, int mode);

unsigned char * LoadDataBuffer(char *filename, int & buf_size);


typedef struct tagBITMAPFILEHEADER2 {
	unsigned short    bfType;
	unsigned long     bfSize;
	unsigned short    bfReserved1;
	unsigned short    bfReserved2;
	unsigned long     bfOffBits;
} BITMAPFILEHEADER2;

typedef struct tag_bitmapinfoheader{
	unsigned long		biSize;
	long				biWidth;
	long				biHeight;
	unsigned short      biPlanes;
	unsigned short      biBitCount;
	unsigned long		biCompression;
	unsigned long		biSizeImage;
	long				biXPelsPerMeter;
	long				biYPelsPerMeter;
	unsigned long		biClrUsed;
	unsigned long		biClrImportant;
} BITMAPINFOHEADER2;

#ifndef WIDTHBYTES
#define WIDTHBYTES(bits)    ((((bits) + 31)>>5)<<2)
#endif

unsigned char *FUNCTION_ReadBmpFile2(const char *filename, int * ht_t, int * wd_t)
{
    BITMAPFILEHEADER2 BmpFileHead;
    BITMAPINFOHEADER2 BmpInfoHead;
    FILE *fp;
    int ht, wd, j, k;
    unsigned char *pImage, *pFileBuf;
    fpos_t filelength = 0;
    int x,y;
    int n;
    
    fp = fopen(filename, "rb");
    if(fp == NULL)
        return NULL;
    
    fseek(fp, 0, SEEK_END);
    fgetpos(fp, &filelength);
    fseek(fp, 0, SEEK_SET);
    fread(&BmpFileHead, 14, 1, fp);
    fread(&BmpInfoHead, 40, 1, fp);
    
    ht = BmpInfoHead.biHeight;
    wd = BmpInfoHead.biWidth;
    
    int linebytes = WIDTHBYTES(wd * BmpInfoHead.biBitCount);
    
    pFileBuf = (unsigned char *)malloc(sizeof(unsigned char)*filelength - 54);
    
    *ht_t = ht;
    *wd_t = wd;
    
    fread(pFileBuf, 1, filelength - 54, fp);
    
    pImage = (unsigned char *)malloc(sizeof(unsigned char)*ht*wd*3);
    for (j= 0; j< ht; j++)
    {
        for(k = 0;k < wd; ++k)
        {
            if(BmpInfoHead.biBitCount > 8)
            {
                y = ht - 1 - j;
                x = k;
                n = y*linebytes + x*3;
                
                unsigned char R, G, B;
                R = pFileBuf[n + 2];
                G = pFileBuf[n + 1];
                B = pFileBuf[n];
                
                //pImage[j*wd + k] = (R*40 + G*75 + B*13)/128;
                pImage[(j*wd + k)*3] = B;
                pImage[(j*wd + k)*3 + 1] = G;
                pImage[(j*wd + k)*3 + 2] = R;
            }
            else
            {
                unsigned char gray;
                
                y = ht - 1 - j;
                x = k;
                n = y*linebytes + x;
                
                gray = pFileBuf[n];
                pImage[j*wd + k] = gray;//GetDibGrayPixel(lpbi, j, k, pImage);
            }
        }
    }
    
    fclose(fp);
    free(pFileBuf);
    return pImage;
}
unsigned char * LoadDataBuffer(char *filename, int & buf_size)
{
    unsigned char *model_buf = NULL;
    fpos_t pos;
    
    FILE *fp;
    fp = fopen(filename, "rb");
    if(!fp)
        return NULL;
    fseek(fp, 0, SEEK_END);
    fgetpos(fp, &pos);
    buf_size = pos;
    model_buf = (unsigned char *) malloc(sizeof(char) * buf_size);
    fseek(fp, 0, SEEK_SET);
    fread(model_buf, sizeof(char), buf_size, fp);
    fclose(fp);
    
    return model_buf;
}
unsigned char FUNCTION_RgbLimit(int input)
{
    unsigned char output;
    output = (unsigned char)((input < 0) ? 0 : ((input > 255) ? 255 : input));
    return output;
}
void FUNCTION_rgb_to_ycc(unsigned char r, unsigned char g, unsigned char b, unsigned char *yp, unsigned char *cb, unsigned char *cr)
{
    int Y, Cb, Cr;
    /*
     fY  = 0.299*r + 0.587*g + 0.114*b;
     fCb = -0.16874*r - 0.33126*g + 0.5*b + 128;
     fCr = 0.5*r - 0.41869*g - 0.08131*b + 128;
     */
    
/*     Y  = (306 * r + 601 * g + 117 * b) >> 10;
    Cb = ((512 * b - 173 * r - 339 * g) + 128 * 1024) >> 10;
    Cr = ((512 * r - 429 * g - 83 * b) + 128 * 1024) >> 10; */
	Y  = (306 * b + 601 * g + 117 * r) >> 10;
    Cb = ((512 * r - 173 * b - 339 * g) + 128 * 1024) >> 10;
    Cr = ((512 * b - 429 * g - 83 * r) + 128 * 1024) >> 10;
    
    *yp = FUNCTION_RgbLimit(Y);
    *cb =FUNCTION_RgbLimit(Cb);
    *cr = FUNCTION_RgbLimit(Cr);
}
void FUNCTION_ycc_to_rgb(unsigned char y, unsigned char cb, unsigned char cr, unsigned char *r, unsigned char *g, unsigned char *b)
{
    //  fR = Y + 1.402*(Cr-128);
    //  fG = Y - 0.34414*(Cb-128) - 0.71414*(Cr-128);
    //  fB = Y + 1.772*(Cb-128);
    
    int tmpR, tmpG, tmpB;
    
    // 10bit shift fixed coding
    tmpR = y - 179 + ((1436 * cr)           >> 10);
    tmpG = y + 135 + ((-352 * cb  - 731 * cr) >> 10);
    tmpB = y - 227 + ((1815 * cb)           >> 10);
    
    *r = FUNCTION_RgbLimit(tmpB);
    *g =FUNCTION_RgbLimit(tmpG);
    *b =FUNCTION_RgbLimit(tmpR);
}

// mode 0 : CbCr, NV12, mode 1 : CrCb, NV21
void FUNCTION_BGR24_to_YUV420SP(unsigned char *pBGR24, int wd, int ht, unsigned char* pYUV420, int mode)
{
    int i, j, ySize;
    
    unsigned char cb, cr;
    unsigned char cb_00, cb_01, cb_10, cb_11;
    unsigned char cr_00, cr_01, cr_10, cr_11;
    unsigned char y_00, y_01, y_10, y_11;
    unsigned char r_00, r_01, r_10, r_11;
    unsigned char g_00, g_01, g_10, g_11;
    unsigned char b_00, b_01, b_10, b_11;
    
    unsigned char  *pTmpY = NULL;
    unsigned short *pTmpC = NULL; // HTH
    unsigned short TmpVal; // HTH
    unsigned char  *pTmpBGR = NULL;
    
    int iOff1, iOff2, jOff1, jOff2;
    
    ySize = wd * ht;
    
    pTmpY = pYUV420;
    pTmpC = (unsigned short*)(pYUV420 + ySize);
    for (i = 0 ; i < (ht >> 1) ; i++)
    {
        iOff1 = (i * wd) << 1; // 2*i*Wd
        iOff2 = iOff1 + wd; // (2*i+1)*Wd
        
        for (j = 0 ; j < (wd >> 1) ; j++)
        {
            jOff1 = (j << 1); // 2*j
            jOff2 = jOff1 + 1; // 2*j + 1
            
            pTmpBGR = pBGR24 + 3 * (iOff1 + jOff1);
            b_00 = *pTmpBGR++;
            g_00 = *pTmpBGR++;
            r_00 = *pTmpBGR ;
            FUNCTION_rgb_to_ycc(r_00, g_00, b_00, &y_00, &cb_00, &cr_00);
            
            pTmpBGR = pBGR24 + 3 * (iOff1 + jOff2);
            b_01 = *pTmpBGR++;
            g_01 = *pTmpBGR++;
            r_01 = *pTmpBGR ;
            FUNCTION_rgb_to_ycc(r_01, g_01, b_01, &y_01, &cb_01, &cr_01);
            
            pTmpBGR = pBGR24 + 3 * (iOff2 + jOff1);
            b_10 = *pTmpBGR++;
            g_10 = *pTmpBGR++;
            r_10 = *pTmpBGR ;
            FUNCTION_rgb_to_ycc(r_10, g_10, b_10, &y_10, &cb_10, &cr_10);
            
            pTmpBGR = pBGR24 + 3 * (iOff2 + jOff2);
            b_11 = *pTmpBGR++;
            g_11 = *pTmpBGR++;
            r_11 = *pTmpBGR ;
            FUNCTION_rgb_to_ycc(r_11, g_11, b_11, &y_11, &cb_11, &cr_11);
            
            pTmpY[iOff1 + jOff1] = y_00;
            pTmpY[iOff1 + jOff2] = y_01;
            pTmpY[iOff2 + jOff1] = y_10;
            pTmpY[iOff2 + jOff2] = y_11;
            
            cb = ((cb_00 + cb_01 + cb_10 + cb_11) >> 2);
            cr = ((cr_00 + cr_01 + cr_10 + cr_11) >> 2);
            
            if (mode)  // YCrCb --> (msb) cb | cr (lsb)
            {
                TmpVal = (unsigned short)cb;
                TmpVal = TmpVal << 8;
                TmpVal = TmpVal + cr;
                //TmpVal = TmpVal | 0xFF;
                //TmpVal = TmpVal & cr;
            }
            else  // YCbCr --> (msb) cr | cb (lsb)
            {
                TmpVal = (unsigned short)cr;
                TmpVal = TmpVal << 8;
                TmpVal = TmpVal + cb;
                //TmpVal = TmpVal | 0xFF;
                //TmpVal = TmpVal & cb;
            }
            *pTmpC++ = TmpVal;
        }
    }
}





// mode 0 : CbCr, NV12, mode 1 : CrCb, NV21
void FUNCTION_YUV420SP_to_BGR24(unsigned char *pYUV420, int wd, int ht, unsigned char* pBGR24, int mode)
{
    int i, j, ySize;
    unsigned char cb, cr;
    unsigned char y_00, y_01, y_10, y_11;
    unsigned char r_00, r_01, r_10, r_11;
    unsigned char g_00, g_01, g_10, g_11;
    unsigned char b_00, b_01, b_10, b_11;
    
    unsigned char  *pTmpY = NULL;
    unsigned short *pTmpC = NULL; // HTH
    unsigned short TmpVal; // HTH
    unsigned char  *pTmpBGR = NULL;
    
    int iOff1, iOff2, jOff1, jOff2;
    
    ySize = wd * ht;
    
    pTmpY = pYUV420;
    pTmpC = (unsigned short*)(pYUV420 + ySize);
    for (i = 0 ; i < (ht >> 1) ; i++)
    {
        
        iOff1 = (i * wd) << 1; // 2*i*Wd
        iOff2 = iOff1 + wd; // (2*i+1)*Wd
        
        for (j = 0 ; j < (wd >> 1) ; j++)
        {
            
            jOff1 = (j << 1); // 2*j
            jOff2 = jOff1 + 1; // 2*j + 1
            
            y_00 = pTmpY[iOff1 + jOff1];
            y_01 = pTmpY[iOff1 + jOff2];
            y_10 = pTmpY[iOff2 + jOff1];
            y_11 = pTmpY[iOff2 + jOff2];
            
            TmpVal = pTmpC[i * (wd >> 1) + j];
            
            if (mode)  // YCrCb
            {
                cr = TmpVal & 0xff;// LSB 8-bit
                cb = (TmpVal >> 8) & 0xff; // MSB 8-bit
            }
            else  // YCbCr
            {
                cb = TmpVal & 0xff;
                cr = (TmpVal >> 8) & 0xff;
            }
            
            FUNCTION_ycc_to_rgb(y_00, cb, cr, &r_00, &g_00, &b_00);
            FUNCTION_ycc_to_rgb(y_01, cb, cr, &r_01, &g_01, &b_01);
            FUNCTION_ycc_to_rgb(y_10, cb, cr, &r_10, &g_10, &b_10);
            FUNCTION_ycc_to_rgb(y_11, cb, cr, &r_11, &g_11, &b_11);
            
            pTmpBGR = pBGR24 + 3 * (iOff1 + jOff1);
            *pTmpBGR++  = b_00;
            *pTmpBGR++  = g_00;
            *pTmpBGR    = r_00;
            
            pTmpBGR = pBGR24 + 3 * (iOff1 + jOff2);
            *pTmpBGR++  = b_01;
            *pTmpBGR++  = g_01;
            *pTmpBGR    = r_01;
            
            pTmpBGR = pBGR24 + 3 * (iOff2 + jOff1);
            *pTmpBGR++  = b_10;
            *pTmpBGR++  = g_10;
            *pTmpBGR    = r_10;
            
            pTmpBGR = pBGR24 + 3 * (iOff2 + jOff2);
            *pTmpBGR++  = b_11;
            *pTmpBGR++  = g_11;
            *pTmpBGR    = r_11;
        }
    }
}


#endif
