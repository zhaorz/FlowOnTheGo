// color_flow.cpp 
// color-code motion field
// normalizes based on specified value, or on maximum motion present otherwise

// DS 2/9/08 fixed bug in MotionToColor concerning reallocation of colim (thanks Yunpeng!)

static char *usage = "\n  usage: %s [-quiet] in.flo out.png [maxmotion]\n";

#include <stdio.h>
#include <math.h>
#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"

int verbose = 1;

void MotionToColor(CFloatImage motim, CByteImage &colim, float maxmotion)
{
    CShape sh = motim.Shape();
    int width = sh.width, height = sh.height;
    colim.ReAllocate(CShape(width, height, 3));
    int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float fx = motim.Pixel(x, y, 0);
	    float fy = motim.Pixel(x, y, 1);
	    if (unknown_flow(fx, fy))
		continue;
	    maxx = __max(maxx, fx);
	    maxy = __max(maxy, fy);
	    minx = __min(minx, fx);
	    miny = __min(miny, fy);
	    float rad = sqrt(fx * fx + fy * fy);
	    maxrad = __max(maxrad, rad);
	}
    }
    printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
	   maxrad, minx, maxx, miny, maxy);


    if (maxmotion > 0) // i.e., specified on commandline
	maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
	maxrad = 1;

    if (verbose)
	fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float fx = motim.Pixel(x, y, 0);
	    float fy = motim.Pixel(x, y, 1);
	    uchar *pix = &colim.Pixel(x, y, 0);
	    if (unknown_flow(fx, fy)) {
		pix[0] = pix[1] = pix[2] = 0;
	    } else {
		computeColor(fx/maxrad, fy/maxrad, pix);
	    }
	}
    }
}

int main(int argc, char *argv[])
{
    try {
	int argn = 1;
	if (argc > 1 && argv[1][0]=='-' && argv[1][1]=='q') {
	    verbose = 0;
	    argn++;
	}
	if (argn >= argc-3 && argn <= argc-2) {
	    char *flowname = argv[argn++];
	    char *outname = argv[argn++];
	    float maxmotion = argn < argc ? atof(argv[argn++]) : -1;
	    CFloatImage im, fband;
	    ReadFlowFile(im, flowname);
	    CByteImage band, outim;
	    CShape sh = im.Shape();
	    sh.nBands = 3;
	    outim.ReAllocate(sh);
	    outim.ClearPixels();
	    MotionToColor(im, outim, maxmotion);
	    WriteImageVerb(outim, outname, verbose);
	} else
	    throw CError(usage, argv[0]);
    }
    catch (CError &err) {
	fprintf(stderr, err.message);
	fprintf(stderr, "\n");
	return -1;
    }

    return 0;
}
