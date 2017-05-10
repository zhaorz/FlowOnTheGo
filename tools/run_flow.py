#! /usr/bin/env python

import os
import subprocess

FLOW_EXE  = "/home/nvidia/private/FlowOnTheGo/src/build/flow"
FRAME_DIR = "/home/nvidia/data/temple_3"
FLOW_DIR  = "/home/nvidia/data/temple_3_flow"
COLOR_EXE = "/home/nvidia/private/FlowOnTheGo/tools/color_flow"
TMP_DIR   = "/home/nvidia/tmp"

def calcFlow(frame0, frame1, output):
    try:
        out = subprocess.check_output([FLOW_EXE, frame0, frame1, output])
        return True
    except subprocess.CalledProcessError as e:
        print e
        return False

def colorFlow(flow, output):
    try:
        out = subprocess.check_output([COLOR_EXE, flow, output])
        return True
    except subprocess.CalledProcessError as e:
        print e
        return False

def processFlow(frame0, frame1):
    tmpOutput = TMP_DIR + os.sep + "flow_out.flo"
    frameNo = frame0.split(os.sep)[-1].split('_')[-1].split('.')[0]
    outFile = FLOW_DIR + os.sep + 'flow_' + frameNo + '.png'

    calcFlow(frame0, frame1, tmpOutput)
    colorFlow(tmpOutput, outFile)


def main():
    frames = sorted(map(lambda fn: FRAME_DIR + os.sep + fn, os.listdir(FRAME_DIR)))
    print("Found %d frames" % len(frames))

    nFrames = len(frames)
    for i in range(0, nFrames - 1):
        frame0 = frames[i]
        frame1 = frames[i+1]

        print("[%2d/%2d] Calculating flow %s => %s" % (i, nFrames, frame0, frame1))
        processFlow(frame0, frame1)

    print "Done"


if __name__ == "__main__":
    main()

