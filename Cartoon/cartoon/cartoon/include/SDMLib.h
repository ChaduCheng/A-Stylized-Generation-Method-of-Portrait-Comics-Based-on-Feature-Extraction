#ifndef SDMLIB_H
#define SDMLIB_H

//#ifdef __cplusplus
//extern "C" {
//#endif

#include "comdef.h"

	MRESULT sdInitialSDEngine(SDEngine* pSDEngine);
	MRESULT sdUninitializeSDEngine(SDEngine* pSDEngine);
	MRESULT sdFaceLandmark(SDEngine pSDEngine, sd_Image inputImg, sd_rect* rect, int rotationAngle, sd_Matrix outLandmarks);

#endif


//#ifdef __cplusplus
//}
//#endif
