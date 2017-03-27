# packages
APT_PKGS := python-pip python-dev
BREW_PKGS := --python
PIP_PKGS := numpy scipy pandas scikit-learn

SED := sed

# directories
DIR_DATA := data
DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_BLEND := $(DIR_BUILD)/blend
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model
DIR_SUB := $(DIR_BUILD)/sub

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_BIN) $(DIR_BLEND) $(DIR_SUB)

# data files for training and predict
DATA_TRN = $(DIR_DATA)/train.csv
DATA_TST = $(DIR_DATA)/test.csv

SAMPLE_SUBMISSION := $(DIR_DATA)/sample_submission.csv

ID_TST := $(DIR_DATA)/id.tst.csv
HEADER := $(DIR_DATA)/header.csv

Y_TRN:= $(DIR_FEATURE)/y.trn.txt
Y_TST:= $(DIR_FEATURE)/y.tst.txt

CV_ID := $(DIR_FEATURE)/cvid.txt

$(DIRS):
	mkdir -p $@

$(HEADER): $(SAMPLE_SUBMISSION)
	head -1 $< > $@

$(ID_TST): $(SAMPLE_SUBMISSION)
	cut -d, -f1 $< | tail -n +2 > $@

$(Y_TST): $(SAMPLE_SUBMISSION) | $(DIR_FEATURE)
	cut -d, -f2 $< | tail -n +2 > $@

$(CV_ID) $(Y_TRN): $(DATA_TRN) | $(DIR_FEATURE)
	python ./src/cal_cv.py --input $< \
                           --cv $(CV_ID) \
                           --ytrn $(Y_TRN)


# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
