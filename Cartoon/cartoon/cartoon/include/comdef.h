#ifndef COMDEF_H
#define COMDEF_H

typedef long					MLong;
typedef float					MFloat;
typedef double					MDouble;
typedef unsigned char			MByte;				
typedef unsigned short			MWord;
typedef unsigned long			MDWord;
typedef void*					MHandle;
typedef char					MChar;
typedef long					MBool;
typedef void					MVoid;
typedef void*					MPVoid;
typedef char*					MPChar;
typedef short					MShort;
typedef const char*				MPCChar;
typedef	MLong					MRESULT;
typedef MDWord					MCOLORREF; 

typedef	signed		char		MInt8;
typedef	unsigned	char		MUInt8;
typedef	signed		short		MInt16;
typedef	unsigned	short		MUInt16;
typedef signed		long		MInt32;
typedef unsigned	long		MUInt32;
typedef unsigned    char        uchar;


#define MOK                0
#define MERR_NO_MEMORY     1
#define MERR_INVALID_PARAM 2
#define MNull		       0
#define MFalse		       0
#define MTrue		       1
#define MIN(v1, v2) 		( ((v1) > (v2)) ? (v2) : (v1) )
#define MAX(v1, v2) 		( ((v1) < (v2)) ? (v2) : (v1) )

typedef struct{
	MLong x;
	MLong y;
	MLong width;
	MLong height;
}sd_rect;

typedef struct{
	MLong width;
	MLong height;
	MLong channels;
	MByte* imageData;
}sd_image;
typedef sd_image* sd_Image;

typedef struct{
	MLong nRow;
	MLong nCol;
	MFloat* data;
}sd_matrix;
typedef sd_matrix* sd_Matrix;

typedef MVoid* SDEngine;


#endif
