EXECUTABLE :=

CU_FILES   :=

CU_DEPS    :=

CC_FILES   :=

LOGS	     :=

BASE_PATH  := $(shell pwd)

REF_EXE    := flow_ref

###########################################################

.PHONY: clean $(REF_EXE)

default:

clean:
		rm -rf build
		rm $(REF_EXE)

$(REF_EXE):
		mkdir -p $(BASE_PATH)/build/$(REF_EXE)
		cmake -B$(BASE_PATH)/build/$(REF_EXE) -H$(BASE_PATH)/ref
		cd $(BASE_PATH)/build/$(REF_EXE)
		make -C $(BASE_PATH)/build/$(REF_EXE)
		mv $(BASE_PATH)/build/$(REF_EXE)/$(REF_EXE) $(BASE_PATH)
