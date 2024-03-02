import subprocess

# list all models that are to be evaluated by this thread

dataset = "J:\\data\\obj"

models = ["ss5", "sm1", "sm2", "sm3", "sm4", "sm5", "sl1", "sl2", "sl3", "sl4", "sl5"]
# 1 THIS ["ss1","ss2","ss3","ss4","ss5","sm1","sm2","sm3","sm4","sm5","sl1","sl2","sl3","sl4","sl5"]
# 2 ["sls1","sls2","sls3","sls4","sls5","rb1","rb2","rb3","rb4","rb5","rba1","rba2","rba3","rba4","rba5"]
# 3 ["rbr1","rbr2","rbr3","rbr4","rbr5","rd1","rd2","rd3","rd4","rd5","rn1","rn2","rn3","rn4","rn5"]
# 4 ["rc1","rc2","rc3","rc4","rc5","ra1","ra2","ra3","ra4","ra5","bs101","bs102","bs103","bs104","bs105"]
# 5 ["bs1001","bs1002","bs1003","bs1004","bs1005","bs10001","bs10002","bs10003","bs10004","bs10005","sb11","sb12","sb13","sb14","sb15"]
# 6 ["sb51","sb52","sb53","sb54","sb55","ts1","ts2","ts3","ts4","ts5","tr1","tr2","tr3","tr4","tr5"]

"""
python .\darknet_evaluation_main.py 
--modelFolder K:\BENCHMARK\TRAINED_MODELS\slstest\backup 
--dataFolder K:\BENCHMARK\REAL\tiny_test 
--darknetFolder K:\darknet\x64 
--configPath K:\BENCHMARK\yolov4_array_HPC_new_test.cfg 
--metaPath K:\BENCHMARK\REAL\all\data\obj.data 
--outputFolder K:\BENCHMARK\ffs
"""

for model in models:
	subprocess.call(
		[
			"python",
			"darknet_evaluation_main.py",
			"--modelFolder",
			"I:\\BENCHMARK\\DARKNET_TRAIN\\NEW_BATCH_LR_0001\\" + model + "\\backup",
			"--dataFolder",
			dataset,
			"--darknetFolder",
			"I:\\BENCHMARK\\DARKNET_TRAIN\\darknet\\x64",
			"--configPath",
			"I:\\BENCHMARK/DARKNET_TRAIN\\yolov4_array_HPC_new_test.cfg",
			"--metaPath",
			"I:\\BENCHMARK/DARKNET_TRAIN\\obj.data",
			"--outputFolder",
			"I:\\BENCHMARK\\DARKNET_TRAIN\\OUTPUT\\" + model,
			"--GPU",
			"0",
		]
	)
